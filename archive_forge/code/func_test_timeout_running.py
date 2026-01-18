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
def test_timeout_running(self):
    self.server.start()
    self.server.stop()
    shutdown_called = eventletutils.Event()

    def slow_shutdown(wait):
        shutdown_called.set()
        eventlet.sleep(10)
    self.executors[0].shutdown = slow_shutdown
    thread = eventlet.spawn(self.server.wait)
    shutdown_called.wait()
    self.assertRaises(server_module.TaskTimeout, self.server.wait, timeout=1)
    thread.kill()