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
class CADFNotificationsForEntities(NotificationsForEntities):

    def setUp(self):
        super(CADFNotificationsForEntities, self).setUp()
        self.config_fixture.config(notification_format='cadf')

    def test_initiator_data_is_set(self):
        ref = unit.new_domain_ref()
        resp = self.post('/domains', body={'domain': ref})
        resource_id = resp.result.get('domain').get('id')
        self._assert_last_audit(resource_id, CREATED_OPERATION, 'domain', cadftaxonomy.SECURITY_DOMAIN)
        self._assert_initiator_data_is_set(CREATED_OPERATION, 'domain', cadftaxonomy.SECURITY_DOMAIN)

    def test_initiator_request_id(self):
        data = self.build_authentication_request(user_id=self.user_id, password=self.user['password'])
        self.post('/auth/tokens', body=data)
        audit = self._audits[-1]
        initiator = audit['payload']['initiator']
        self.assertIn('request_id', initiator)

    def test_initiator_global_request_id(self):
        global_request_id = 'req-%s' % uuid.uuid4()
        data = self.build_authentication_request(user_id=self.user_id, password=self.user['password'])
        self.post('/auth/tokens', body=data, headers={'X-OpenStack-Request-Id': global_request_id})
        audit = self._audits[-1]
        initiator = audit['payload']['initiator']
        self.assertEqual(initiator['global_request_id'], global_request_id)

    def test_initiator_global_request_id_not_set(self):
        data = self.build_authentication_request(user_id=self.user_id, password=self.user['password'])
        self.post('/auth/tokens', body=data)
        audit = self._audits[-1]
        initiator = audit['payload']['initiator']
        self.assertNotIn('global_request_id', initiator)