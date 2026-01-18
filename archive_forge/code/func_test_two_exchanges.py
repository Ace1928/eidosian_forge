import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_two_exchanges(self):
    transport = msg_notifier.get_notification_transport(self.conf, url='fake:')
    endpoint = mock.Mock()
    endpoint.info.return_value = None
    targets = [oslo_messaging.Target(topic='topic', exchange='exchange1'), oslo_messaging.Target(topic='topic', exchange='exchange2')]
    listener_thread = self._setup_listener(transport, [endpoint], targets=targets)
    notifier = self._setup_notifier(transport, topics=['topic'])

    def mock_notifier_exchange(name):

        def side_effect(target, ctxt, message, version, retry):
            target.exchange = name
            return transport._driver.send_notification(target, ctxt, message, version, retry=retry)
        transport._send_notification = mock.MagicMock(side_effect=side_effect)
    notifier.info(test_utils.TestContext(user_name='bob0'), 'an_event.start', 'test message default exchange')
    mock_notifier_exchange('exchange1')
    ctxt1 = test_utils.TestContext(user_name='bob1')
    notifier.info(ctxt1, 'an_event.start', 'test message exchange1')
    mock_notifier_exchange('exchange2')
    ctxt2 = test_utils.TestContext(user_name='bob2')
    notifier.info(ctxt2, 'an_event.start', 'test message exchange2')
    self.wait_for_messages(2)
    self.assertFalse(listener_thread.stop())
    endpoint.info.assert_has_calls([mock.call(ctxt1, 'testpublisher', 'an_event.start', 'test message exchange1', {'timestamp': mock.ANY, 'message_id': mock.ANY}), mock.call(ctxt2, 'testpublisher', 'an_event.start', 'test message exchange2', {'timestamp': mock.ANY, 'message_id': mock.ANY})], any_order=True)