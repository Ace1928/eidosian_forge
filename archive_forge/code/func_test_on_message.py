from taskflow.engines.worker_based import dispatcher
from taskflow import test
from taskflow.test import mock
def test_on_message(self):
    on_hello = mock.MagicMock()
    handlers = {'hello': dispatcher.Handler(on_hello)}
    d = dispatcher.TypeDispatcher(type_handlers=handlers)
    msg = mock_acked_message(properties={'type': 'hello'})
    d.on_message('', msg)
    self.assertTrue(on_hello.called)
    self.assertTrue(msg.ack_log_error.called)
    self.assertTrue(msg.acknowledged)