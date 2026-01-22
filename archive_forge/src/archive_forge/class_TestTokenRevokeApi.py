import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class TestTokenRevokeApi(TestTokenRevokeById):
    """Test token revocation on the v3 Identity API."""

    def config_overrides(self):
        super(TestTokenRevokeApi, self).config_overrides()
        self.config_fixture.config(group='token', provider='fernet', revoke_by_id=False)
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))

    def assertValidDeletedProjectResponse(self, events_response, project_id):
        events = events_response['events']
        self.assertEqual(1, len(events))
        self.assertEqual(project_id, events[0]['project_id'])
        self.assertIsNotNone(events[0]['issued_before'])
        self.assertIsNotNone(events_response['links'])
        del events_response['events'][0]['issued_before']
        del events_response['events'][0]['revoked_at']
        del events_response['links']
        expected_response = {'events': [{'project_id': project_id}]}
        self.assertEqual(expected_response, events_response)

    def assertValidRevokedTokenResponse(self, events_response, **kwargs):
        events = events_response['events']
        self.assertEqual(1, len(events))
        for k, v in kwargs.items():
            self.assertEqual(v, events[0].get(k))
        self.assertIsNotNone(events[0]['issued_before'])
        self.assertIsNotNone(events_response['links'])
        del events_response['events'][0]['issued_before']
        del events_response['events'][0]['revoked_at']
        del events_response['links']
        expected_response = {'events': [kwargs]}
        self.assertEqual(expected_response, events_response)

    def test_revoke_token(self):
        scoped_token = self.get_scoped_token()
        headers = {'X-Subject-Token': scoped_token}
        response = self.get('/auth/tokens', headers=headers).json_body['token']
        self.delete('/auth/tokens', headers=headers)
        self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)
        events_response = self.get('/OS-REVOKE/events').json_body
        self.assertValidRevokedTokenResponse(events_response, audit_id=response['audit_ids'][0])

    def test_get_revoke_by_id_false_returns_gone(self):
        self.get('/auth/tokens/OS-PKI/revoked', expected_status=http.client.GONE)

    def test_head_revoke_by_id_false_returns_gone(self):
        self.head('/auth/tokens/OS-PKI/revoked', expected_status=http.client.GONE)

    def test_revoke_by_id_true_returns_forbidden(self):
        self.config_fixture.config(group='token', revoke_by_id=True)
        self.get('/auth/tokens/OS-PKI/revoked', expected_status=http.client.FORBIDDEN)
        self.head('/auth/tokens/OS-PKI/revoked', expected_status=http.client.FORBIDDEN)

    def test_list_delete_project_shows_in_event_list(self):
        self.role_data_fixtures()
        events = self.get('/OS-REVOKE/events').json_body['events']
        self.assertEqual([], events)
        self.delete('/projects/%(project_id)s' % {'project_id': self.projectA['id']})
        events_response = self.get('/OS-REVOKE/events').json_body
        self.assertValidDeletedProjectResponse(events_response, self.projectA['id'])

    def assertEventDataInList(self, events, **kwargs):
        found = False
        for e in events:
            for key, value in kwargs.items():
                try:
                    if e[key] != value:
                        break
                except KeyError:
                    break
            else:
                found = True
        self.assertTrue(found, 'event with correct values not in list, expected to find event with key-value pairs. Expected: "%(expected)s" Events: "%(events)s"' % {'expected': ','.join(["'%s=%s'" % (k, v) for k, v in kwargs.items()]), 'events': events})

    def test_list_delete_token_shows_in_event_list(self):
        self.role_data_fixtures()
        events = self.get('/OS-REVOKE/events').json_body['events']
        self.assertEqual([], events)
        scoped_token = self.get_scoped_token()
        headers = {'X-Subject-Token': scoped_token}
        auth_req = self.build_authentication_request(token=scoped_token)
        response = self.v3_create_token(auth_req)
        token2 = response.json_body['token']
        headers2 = {'X-Subject-Token': response.headers['X-Subject-Token']}
        response = self.v3_create_token(auth_req)
        response.json_body['token']
        headers3 = {'X-Subject-Token': response.headers['X-Subject-Token']}
        self.head('/auth/tokens', headers=headers, expected_status=http.client.OK)
        self.head('/auth/tokens', headers=headers2, expected_status=http.client.OK)
        self.head('/auth/tokens', headers=headers3, expected_status=http.client.OK)
        self.delete('/auth/tokens', headers=headers)
        events_response = self.get('/OS-REVOKE/events').json_body
        events = events_response['events']
        self.assertEqual(1, len(events))
        self.assertEventDataInList(events, audit_id=token2['audit_ids'][1])
        self.head('/auth/tokens', headers=headers, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers=headers2, expected_status=http.client.OK)
        self.head('/auth/tokens', headers=headers3, expected_status=http.client.OK)

    def test_list_with_filter(self):
        self.role_data_fixtures()
        events = self.get('/OS-REVOKE/events').json_body['events']
        self.assertEqual(0, len(events))
        scoped_token = self.get_scoped_token()
        headers = {'X-Subject-Token': scoped_token}
        auth = self.build_authentication_request(token=scoped_token)
        headers2 = {'X-Subject-Token': self.get_requested_token(auth)}
        self.delete('/auth/tokens', headers=headers)
        self.delete('/auth/tokens', headers=headers2)
        events = self.get('/OS-REVOKE/events').json_body['events']
        self.assertEqual(2, len(events))
        future = utils.isotime(timeutils.utcnow() + datetime.timedelta(seconds=1000))
        events = self.get('/OS-REVOKE/events?since=%s' % future).json_body['events']
        self.assertEqual(0, len(events))