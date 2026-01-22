import datetime
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
from keystone.models import revoke_model
from keystone.tests.unit import test_v3
class OSRevokeTests(test_v3.RestfulTestCase, test_v3.JsonHomeTestMixin):
    JSON_HOME_DATA = {'https://docs.openstack.org/api/openstack-identity/3/ext/OS-REVOKE/1.0/rel/events': {'href': '/OS-REVOKE/events'}}

    def test_get_empty_list(self):
        resp = self.get('/OS-REVOKE/events')
        self.assertEqual([], resp.json_body['events'])

    def _blank_event(self):
        return {}

    def assertReportedEventMatchesRecorded(self, event, sample, before_time):
        after_time = timeutils.utcnow()
        event_issued_before = timeutils.normalize_time(timeutils.parse_isotime(event['issued_before']))
        self.assertLessEqual(before_time, event_issued_before, 'invalid event issued_before time; %s is not later than %s.' % (utils.isotime(event_issued_before, subsecond=True), utils.isotime(before_time, subsecond=True)))
        self.assertLessEqual(event_issued_before, after_time, 'invalid event issued_before time; %s is not earlier than %s.' % (utils.isotime(event_issued_before, subsecond=True), utils.isotime(after_time, subsecond=True)))
        del event['issued_before']
        del event['revoked_at']
        self.assertEqual(sample, event)

    def test_revoked_list_self_url(self):
        revoked_list_url = '/OS-REVOKE/events'
        resp = self.get(revoked_list_url)
        links = resp.json_body['links']
        self.assertThat(links['self'], matchers.EndsWith(revoked_list_url))

    def test_revoked_token_in_list(self):
        audit_id = uuid.uuid4().hex
        sample = self._blank_event()
        sample['audit_id'] = str(audit_id)
        before_time = timeutils.utcnow().replace(microsecond=0)
        PROVIDERS.revoke_api.revoke_by_audit_id(audit_id)
        resp = self.get('/OS-REVOKE/events')
        events = resp.json_body['events']
        self.assertEqual(1, len(events))
        self.assertReportedEventMatchesRecorded(events[0], sample, before_time)

    def test_disabled_project_in_list(self):
        project_id = uuid.uuid4().hex
        sample = dict()
        sample['project_id'] = str(project_id)
        before_time = timeutils.utcnow().replace(microsecond=0)
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(project_id=project_id))
        resp = self.get('/OS-REVOKE/events')
        events = resp.json_body['events']
        self.assertEqual(1, len(events))
        self.assertReportedEventMatchesRecorded(events[0], sample, before_time)

    def test_disabled_domain_in_list(self):
        domain_id = uuid.uuid4().hex
        sample = dict()
        sample['domain_id'] = str(domain_id)
        before_time = timeutils.utcnow().replace(microsecond=0)
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(domain_id=domain_id))
        resp = self.get('/OS-REVOKE/events')
        events = resp.json_body['events']
        self.assertEqual(1, len(events))
        self.assertReportedEventMatchesRecorded(events[0], sample, before_time)

    def test_list_since_invalid(self):
        self.get('/OS-REVOKE/events?since=blah', expected_status=http.client.BAD_REQUEST)

    def test_list_since_valid(self):
        resp = self.get('/OS-REVOKE/events?since=2013-02-27T18:30:59.999999Z')
        events = resp.json_body['events']
        self.assertEqual(0, len(events))

    def test_since_future_time_no_events(self):
        domain_id = uuid.uuid4().hex
        sample = dict()
        sample['domain_id'] = str(domain_id)
        PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(domain_id=domain_id))
        resp = self.get('/OS-REVOKE/events')
        events = resp.json_body['events']
        self.assertEqual(1, len(events))
        resp = self.get('/OS-REVOKE/events?since=%s' % _future_time_string())
        events = resp.json_body['events']
        self.assertEqual([], events)

    def test_revoked_at_in_list(self):
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            revoked_at = timeutils.utcnow()
            PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(revoked_at=revoked_at))
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            resp = self.get('/OS-REVOKE/events')
            events = resp.json_body['events']
            self.assertThat(events, matchers.HasLength(1))
            self.assertTimestampEqual(utils.isotime(revoked_at), events[0]['revoked_at'])

    def test_access_token_id_not_in_event(self):
        ref = {'description': uuid.uuid4().hex}
        resp = self.post('/OS-OAUTH1/consumers', body={'consumer': ref})
        consumer_id = resp.result['consumer']['id']
        PROVIDERS.oauth_api.delete_consumer(consumer_id)
        resp = self.get('/OS-REVOKE/events')
        events = resp.json_body['events']
        self.assertThat(events, matchers.HasLength(1))
        event = events[0]
        self.assertEqual(consumer_id, event['OS-OAUTH1:consumer_id'])
        self.assertNotIn('OS-OAUTH1:access_token_id', event)

    def test_retries_on_deadlock(self):
        patcher = mock.patch('sqlalchemy.orm.query.Query.delete', autospec=True)

        class FakeDeadlock(object):

            def __init__(self, mock_patcher):
                self.deadlock_count = 2
                self.mock_patcher = mock_patcher
                self.patched = True

            def __call__(self, *args, **kwargs):
                if self.deadlock_count > 1:
                    self.deadlock_count -= 1
                else:
                    self.mock_patcher.stop()
                    self.patched = False
                raise oslo_db_exception.DBDeadlock
        sql_delete_mock = patcher.start()
        side_effect = FakeDeadlock(patcher)
        sql_delete_mock.side_effect = side_effect
        try:
            PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(user_id=uuid.uuid4().hex))
        finally:
            if side_effect.patched:
                patcher.stop()
        call_count = sql_delete_mock.call_count
        revoke_attempt_count = 2
        self.assertEqual(call_count, revoke_attempt_count)