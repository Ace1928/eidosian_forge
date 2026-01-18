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
def test_receive_identityId_from_audit_notification(self):
    observer = None
    resource_type = EXP_RESOURCE_TYPE
    ref = getattr(self, 'service', None)
    if ref is None or ref['type'] != 'identity':
        ref = unit.new_service_ref()
        ref['type'] = 'identity'
        PROVIDERS.catalog_api.create_service(ref['id'], ref.copy())
    action = CREATED_OPERATION + '.' + resource_type
    initiator = notifications._get_request_audit_info(self.user_id)
    target = cadfresource.Resource(typeURI=cadftaxonomy.ACCOUNT_USER)
    outcome = 'success'
    event_type = 'identity.authenticate.created'
    with mock.patch.object(notifications._get_notifier(), 'info') as mocked:
        notifications._send_audit_notification(action, initiator, outcome, target, event_type)
        for mock_args_list in mocked.call_args:
            if len(mock_args_list) != 0:
                for mock_args in mock_args_list:
                    if 'observer' in mock_args:
                        observer = mock_args['observer']
                        break
    self.assertEqual(ref['id'], observer['id'])