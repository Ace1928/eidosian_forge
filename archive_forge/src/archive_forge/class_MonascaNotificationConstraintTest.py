from unittest import mock
from heat.common import exception as heat_exception
from heat.engine.clients.os import monasca as client_plugin
from heat.tests import common
from heat.tests import utils
class MonascaNotificationConstraintTest(common.HeatTestCase):

    def test_expected_exceptions(self):
        self.assertEqual((heat_exception.EntityNotFound,), client_plugin.MonascaNotificationConstraint.expected_exceptions, 'MonascaNotificationConstraint expected exceptions error')

    def test_constraint(self):
        constraint = client_plugin.MonascaNotificationConstraint()
        client_mock = mock.MagicMock()
        client_plugin_mock = mock.MagicMock()
        client_plugin_mock.get_notification.return_value = None
        client_mock.client_plugin.return_value = client_plugin_mock
        self.assertIsNone(constraint.validate_with_client(client_mock, 'notification_1'))
        client_plugin_mock.get_notification.assert_called_once_with('notification_1')