from taskflow.engines.worker_based import dispatcher
from taskflow import test
from taskflow.test import mock
def test_on_requeue_message(self):
    d = dispatcher.TypeDispatcher()
    d.requeue_filters.append(lambda data, message: True)
    msg = mock_acked_message()
    d.on_message('', msg)
    self.assertTrue(msg.requeue.called)
    self.assertFalse(msg.acknowledged)