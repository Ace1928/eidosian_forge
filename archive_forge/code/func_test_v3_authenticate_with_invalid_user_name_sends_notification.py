import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_v3_authenticate_with_invalid_user_name_sends_notification(self):
    user_name = uuid.uuid4().hex
    password = self.user['password']
    domain_id = self.domain_id
    data = self.build_authentication_request(username=user_name, user_domain_id=domain_id, password=password)
    self.post('/auth/tokens', body=data, expected_status=http.client.UNAUTHORIZED)
    note = self._get_last_note()
    initiator = note['initiator']
    self.assertEqual(self.ACTION, note['action'])
    self.assertEqual(user_name, initiator.user_name)
    self.assertEqual(domain_id, initiator.domain_id)
    self.assertTrue(note['send_notification_called'])
    self.assertEqual(cadftaxonomy.OUTCOME_FAILURE, note['event'].outcome)
    self.assertEqual(self.LOCAL_HOST, initiator.host.address)