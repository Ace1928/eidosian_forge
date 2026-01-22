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
class NotificationsTestCase(unit.BaseTestCase):

    def setUp(self):
        super(NotificationsTestCase, self).setUp()
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        self.config_fixture.config(group='oslo_messaging_notifications', transport_url='rabbit://')

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

    def test_send_notification_with_opt_out(self):
        """Test the private method _send_notification with opt-out.

        Test that _send_notification does not notify when a valid
        notification_opt_out configuration is provided.
        """
        resource = uuid.uuid4().hex
        resource_type = EXP_RESOURCE_TYPE
        operation = CREATED_OPERATION
        event_type = 'identity.%s.created' % resource_type
        conf = self.useFixture(config_fixture.Config(CONF))
        conf.config(notification_opt_out=[event_type])
        with mock.patch.object(notifications._get_notifier(), 'info') as mocked:
            notifications._send_notification(operation, resource_type, resource)
            mocked.assert_not_called()

    def test_send_audit_notification_with_opt_out(self):
        """Test the private method _send_audit_notification with opt-out.

        Test that _send_audit_notification does not notify when a valid
        notification_opt_out configuration is provided.
        """
        resource_type = EXP_RESOURCE_TYPE
        action = CREATED_OPERATION + '.' + resource_type
        initiator = mock
        target = mock
        outcome = 'success'
        event_type = 'identity.%s.created' % resource_type
        conf = self.useFixture(config_fixture.Config(CONF))
        conf.config(notification_opt_out=[event_type])
        with mock.patch.object(notifications._get_notifier(), 'info') as mocked:
            notifications._send_audit_notification(action, initiator, outcome, target, event_type)
            mocked.assert_not_called()

    def test_opt_out_authenticate_event(self):
        """Test that authenticate events are successfully opted out."""
        resource_type = EXP_RESOURCE_TYPE
        action = CREATED_OPERATION + '.' + resource_type
        initiator = mock
        target = mock
        outcome = 'success'
        event_type = 'identity.authenticate'
        meter_name = '%s.%s' % (event_type, outcome)
        conf = self.useFixture(config_fixture.Config(CONF))
        conf.config(notification_opt_out=[meter_name])
        with mock.patch.object(notifications._get_notifier(), 'info') as mocked:
            notifications._send_audit_notification(action, initiator, outcome, target, event_type)
            mocked.assert_not_called()