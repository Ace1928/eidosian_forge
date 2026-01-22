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
def _test_no_client_topic(self, call=True):
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    client = self._setup_client(transport, topic=None)
    method = client.call if call else client.cast
    try:
        method({}, 'ping', arg='foo')
    except Exception as ex:
        self.assertIsInstance(ex, oslo_messaging.InvalidTarget, ex)
        self.assertIsNotNone(ex.target)
    else:
        self.assertTrue(False)