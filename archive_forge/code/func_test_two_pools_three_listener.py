import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_two_pools_three_listener(self):
    transport = msg_notifier.get_notification_transport(self.conf, url='fake:')
    endpoint1 = mock.Mock()
    endpoint1.info.return_value = None
    endpoint2 = mock.Mock()
    endpoint2.info.return_value = None
    endpoint3 = mock.Mock()
    endpoint3.info.return_value = None
    targets = [oslo_messaging.Target(topic='topic')]
    listener1_thread = self._setup_listener(transport, [endpoint1], targets=targets, pool='pool1')
    listener2_thread = self._setup_listener(transport, [endpoint2], targets=targets, pool='pool2')
    listener3_thread = self._setup_listener(transport, [endpoint3], targets=targets, pool='pool2')

    def mocked_endpoint_call(i, ctxt):
        return mock.call(ctxt, 'testpublisher', 'an_event.start', 'test message%d' % i, {'timestamp': mock.ANY, 'message_id': mock.ANY})
    notifier = self._setup_notifier(transport, topics=['topic'])
    mocked_endpoint1_calls = []
    for i in range(0, 25):
        ctxt = test_utils.TestContext(user_name='bob%d' % i)
        notifier.info(ctxt, 'an_event.start', 'test message%d' % i)
        mocked_endpoint1_calls.append(mocked_endpoint_call(i, ctxt))
    self.wait_for_messages(25, 'pool2')
    listener2_thread.stop()
    for i in range(0, 25):
        cxt = test_utils.TestContext(user_name='bob%d' % i)
        notifier.info(cxt, 'an_event.start', 'test message%d' % i)
        mocked_endpoint1_calls.append(mocked_endpoint_call(i, cxt))
    self.wait_for_messages(50, 'pool2')
    listener2_thread.start()
    listener3_thread.stop()
    for i in range(0, 25):
        ctxt = test_utils.TestContext(user_name='bob%d' % i)
        notifier.info(ctxt, 'an_event.start', 'test message%d' % i)
        mocked_endpoint1_calls.append(mocked_endpoint_call(i, ctxt))
    self.wait_for_messages(75, 'pool2')
    listener3_thread.start()
    for i in range(0, 25):
        ctxt = test_utils.TestContext(user_name='bob%d' % i)
        notifier.info(ctxt, 'an_event.start', 'test message%d' % i)
        mocked_endpoint1_calls.append(mocked_endpoint_call(i, ctxt))
    self.wait_for_messages(100, 'pool1')
    self.wait_for_messages(100, 'pool2')
    self.assertFalse(listener3_thread.stop())
    self.assertFalse(listener2_thread.stop())
    self.assertFalse(listener1_thread.stop())
    self.assertEqual(100, endpoint1.info.call_count)
    endpoint1.info.assert_has_calls(mocked_endpoint1_calls)
    self.assertLessEqual(25, endpoint2.info.call_count)
    self.assertLessEqual(25, endpoint3.info.call_count)
    self.assertEqual(100, endpoint2.info.call_count + endpoint3.info.call_count)
    for call in mocked_endpoint1_calls:
        self.assertIn(call, endpoint2.info.mock_calls + endpoint3.info.mock_calls)