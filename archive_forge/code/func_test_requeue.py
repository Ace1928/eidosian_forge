import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_requeue(self):
    transport = msg_notifier.get_notification_transport(self.conf, url='fake:')
    endpoint = mock.Mock()
    endpoint.info = mock.Mock()

    def side_effect_requeue(*args, **kwargs):
        if endpoint.info.call_count == 1:
            return oslo_messaging.NotificationResult.REQUEUE
        return oslo_messaging.NotificationResult.HANDLED
    endpoint.info.side_effect = side_effect_requeue
    listener_thread = self._setup_listener(transport, [endpoint])
    notifier = self._setup_notifier(transport)
    cxt = test_utils.TestContext()
    notifier.info(cxt, 'an_event.start', 'test')
    self.wait_for_messages(2)
    self.assertFalse(listener_thread.stop())
    endpoint.info.assert_has_calls([mock.call(cxt, 'testpublisher', 'an_event.start', 'test', {'timestamp': mock.ANY, 'message_id': mock.ANY}), mock.call(cxt, 'testpublisher', 'an_event.start', 'test', {'timestamp': mock.ANY, 'message_id': mock.ANY})])