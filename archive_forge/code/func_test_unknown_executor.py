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
def test_unknown_executor(self):
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    try:
        oslo_messaging.get_rpc_server(transport, None, [], executor='foo')
    except Exception as ex:
        self.assertIsInstance(ex, oslo_messaging.ExecutorLoadFailure)
        self.assertEqual('foo', ex.executor)
    else:
        self.assertTrue(False)