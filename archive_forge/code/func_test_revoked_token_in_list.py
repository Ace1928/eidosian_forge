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