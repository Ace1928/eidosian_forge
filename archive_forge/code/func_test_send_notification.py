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
def test_send_notification(self):
    """Test _send_notification.

        Test the private method _send_notification to ensure event_type,
        payload, and context are built and passed properly.

        """
    resource = uuid.uuid4().hex
    resource_type = EXP_RESOURCE_TYPE
    operation = CREATED_OPERATION
    conf = self.useFixture(config_fixture.Config(CONF))
    conf.config(notification_format='basic')
    expected_args = [{}, 'identity.%s.created' % resource_type, {'resource_info': resource}]
    with mock.patch.object(notifications._get_notifier(), 'info') as mocked:
        notifications._send_notification(operation, resource_type, resource)
        mocked.assert_called_once_with(*expected_args)