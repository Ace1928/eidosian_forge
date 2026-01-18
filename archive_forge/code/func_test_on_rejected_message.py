from taskflow.engines.worker_based import dispatcher
from taskflow import test
from taskflow.test import mock
def test_on_rejected_message(self):
    d = dispatcher.TypeDispatcher()
    msg = mock_acked_message(properties={'type': 'hello'})
    d.on_message('', msg)
    self.assertTrue(msg.reject_log_error.called)
    self.assertFalse(msg.acknowledged)