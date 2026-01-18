import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
def test_state_wrapping(self):
    complete_event = eventletutils.Event()
    complete_waiting_callback = eventletutils.Event()
    start_state = self.server._states['start']
    old_wait_for_completion = start_state.wait_for_completion
    waited = [False]

    def new_wait_for_completion(*args, **kwargs):
        if not waited[0]:
            waited[0] = True
            complete_waiting_callback.set()
            complete_event.wait()
        old_wait_for_completion(*args, **kwargs)
    start_state.wait_for_completion = new_wait_for_completion
    thread1 = eventlet.spawn(self.server.stop)
    thread1_finished = eventletutils.Event()
    thread1.link(lambda _: thread1_finished.set())
    self.server.start()
    complete_waiting_callback.wait()
    self.assertEqual(1, len(self.executors))
    self.assertEqual([], self.executors[0]._calls)
    self.assertFalse(thread1_finished.is_set())
    self.server.stop()
    self.server.wait()
    self.assertEqual(1, len(self.executors))
    self.assertEqual(['shutdown'], self.executors[0]._calls)
    self.assertFalse(thread1_finished.is_set())
    self.server.start()
    self.assertEqual(2, len(self.executors))
    self.assertEqual(['shutdown'], self.executors[0]._calls)
    self.assertEqual([], self.executors[1]._calls)
    self.assertFalse(thread1_finished.is_set())
    complete_event.set()
    thread1_finished.wait()
    self.assertEqual(2, len(self.executors))
    self.assertEqual(['shutdown'], self.executors[0]._calls)
    self.assertEqual([], self.executors[1]._calls)
    self.assertTrue(thread1_finished.is_set())