import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_two_pools(self):
    transport = msg_notifier.get_notification_transport(self.conf, url='fake:')
    endpoint1 = mock.Mock()
    endpoint1.info.return_value = None
    endpoint2 = mock.Mock()
    endpoint2.info.return_value = None
    targets = [oslo_messaging.Target(topic='topic')]
    listener1_thread = self._setup_listener(transport, [endpoint1], targets=targets, pool='pool1')
    listener2_thread = self._setup_listener(transport, [endpoint2], targets=targets, pool='pool2')
    notifier = self._setup_notifier(transport, topics=['topic'])
    ctxts = [test_utils.TestContext(user_name='bob0'), test_utils.TestContext(user_name='bob1')]
    notifier.info(ctxts[0], 'an_event.start', 'test message0')
    notifier.info(ctxts[1], 'an_event.start', 'test message1')
    self.wait_for_messages(2, 'pool1')
    self.wait_for_messages(2, 'pool2')
    self.assertFalse(listener2_thread.stop())
    self.assertFalse(listener1_thread.stop())

    def mocked_endpoint_call(i, ctxts):
        return mock.call(ctxts[i], 'testpublisher', 'an_event.start', 'test message%d' % i, {'timestamp': mock.ANY, 'message_id': mock.ANY})
    endpoint1.info.assert_has_calls([mocked_endpoint_call(0, ctxts), mocked_endpoint_call(1, ctxts)])
    endpoint2.info.assert_has_calls([mocked_endpoint_call(0, ctxts), mocked_endpoint_call(1, ctxts)])