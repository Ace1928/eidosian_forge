import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
def test_two_topics(self):
    transport = msg_notifier.get_notification_transport(self.conf, url='fake:')
    endpoint = mock.Mock()
    endpoint.info.return_value = None
    targets = [oslo_messaging.Target(topic='topic1'), oslo_messaging.Target(topic='topic2')]
    listener_thread = self._setup_listener(transport, [endpoint], targets=targets)
    notifier = self._setup_notifier(transport, topics=['topic1'])
    cxt1 = test_utils.TestContext(user_name='bob')
    notifier.info(cxt1, 'an_event.start1', 'test')
    notifier = self._setup_notifier(transport, topics=['topic2'])
    cxt2 = test_utils.TestContext(user_name='bob2')
    notifier.info(cxt2, 'an_event.start2', 'test')
    self.wait_for_messages(2)
    self.assertFalse(listener_thread.stop())
    endpoint.info.assert_has_calls([mock.call(cxt1, 'testpublisher', 'an_event.start1', 'test', {'timestamp': mock.ANY, 'message_id': mock.ANY}), mock.call(cxt2, 'testpublisher', 'an_event.start2', 'test', {'timestamp': mock.ANY, 'message_id': mock.ANY})], any_order=True)