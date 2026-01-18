import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_two_endpoints(self):
    transport = msg_notifier.get_notification_transport(self.conf, url='fake:')
    endpoint1 = mock.Mock()
    endpoint1.info.return_value = None
    endpoint2 = mock.Mock()
    endpoint2.info.return_value = oslo_messaging.NotificationResult.HANDLED
    listener_thread = self._setup_listener(transport, [endpoint1, endpoint2])
    notifier = self._setup_notifier(transport)
    cxt = test_utils.TestContext()
    notifier.info(cxt, 'an_event.start', 'test')
    self.wait_for_messages(1)
    self.assertFalse(listener_thread.stop())
    endpoint1.info.assert_called_once_with(cxt, 'testpublisher', 'an_event.start', 'test', {'timestamp': mock.ANY, 'message_id': mock.ANY})
    endpoint2.info.assert_called_once_with(cxt, 'testpublisher', 'an_event.start', 'test', {'timestamp': mock.ANY, 'message_id': mock.ANY})