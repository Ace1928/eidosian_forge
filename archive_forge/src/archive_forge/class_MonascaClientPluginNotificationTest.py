from unittest import mock
from heat.common import exception as heat_exception
from heat.engine.clients.os import monasca as client_plugin
from heat.tests import common
from heat.tests import utils
class MonascaClientPluginNotificationTest(common.HeatTestCase):
    sample_uuid = '477e8273-60a7-4c41-b683-fdb0bc7cd152'
    sample_name = 'test-notification'

    def _get_mock_notification(self):
        notification = dict()
        notification['id'] = self.sample_uuid
        notification['name'] = self.sample_name
        return notification

    def setUp(self):
        super(MonascaClientPluginNotificationTest, self).setUp()
        self._client = mock.MagicMock()
        self.client_plugin = client_plugin.MonascaClientPlugin(context=mock.MagicMock())

    @mock.patch.object(client_plugin.MonascaClientPlugin, 'client')
    def test_get_notification(self, client_monasca):
        mock_notification = self._get_mock_notification()
        self._client.notifications.get.return_value = mock_notification
        client_monasca.return_value = self._client
        self.assertEqual(self.sample_uuid, self.client_plugin.get_notification(self.sample_uuid))
        self._client.notifications.get.assert_called_once_with(notification_id=self.sample_uuid)

    @mock.patch.object(client_plugin.MonascaClientPlugin, 'client')
    def test_get_notification_not_found(self, client_monasca):
        self._client.notifications.get.side_effect = client_plugin.monasca_exc.NotFound
        client_monasca.return_value = self._client
        ex = self.assertRaises(heat_exception.EntityNotFound, self.client_plugin.get_notification, self.sample_uuid)
        msg = 'The Monasca Notification (%(name)s) could not be found.' % {'name': self.sample_uuid}
        self.assertEqual(msg, str(ex))
        self._client.notifications.get.assert_called_once_with(notification_id=self.sample_uuid)