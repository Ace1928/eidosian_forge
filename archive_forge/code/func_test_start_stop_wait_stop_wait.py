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
def test_start_stop_wait_stop_wait(self):
    self.server.start()
    self.server.stop()
    self.server.wait()
    self.server.stop()
    self.server.wait()
    self.assertEqual(len(self.executors), 1)
    self.assertEqual(['shutdown'], self.executors[0]._calls)
    self.assertTrue(self.server.listener.cleanup.called)