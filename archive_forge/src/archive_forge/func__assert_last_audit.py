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
def _assert_last_audit(self, resource_id, operation, resource_type, target_uri, reason=None):
    if CONF.notification_format != 'cadf':
        return
    self.assertGreater(len(self._audits), 0)
    audit = self._audits[-1]
    payload = audit['payload']
    if 'resource_info' in payload:
        self.assertEqual(resource_id, payload['resource_info'])
    action = '.'.join(filter(None, [operation, resource_type]))
    self.assertEqual(action, payload['action'])
    self.assertEqual(target_uri, payload['target']['typeURI'])
    if resource_id:
        self.assertEqual(resource_id, payload['target']['id'])
    event_type = '.'.join(filter(None, ['identity', resource_type, operation]))
    self.assertEqual(event_type, audit['event_type'])
    if reason:
        self.assertEqual(reason['reasonCode'], payload['reason']['reasonCode'])
        self.assertEqual(reason['reasonType'], payload['reason']['reasonType'])
    self.assertTrue(audit['send_notification_called'])