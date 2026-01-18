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
def test_server_wait_method(self):
    transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
    target = oslo_messaging.Target(topic='foo', server='bar')
    endpoints = [object()]
    serializer = object()

    class MagicMockIgnoreArgs(mock.MagicMock):
        """MagicMock ignores arguments.

            A MagicMock which can never misinterpret the arguments passed to
            it during construction.
            """

        def __init__(self, *args, **kwargs):
            super(MagicMockIgnoreArgs, self).__init__()
    server = oslo_messaging.get_rpc_server(transport, target, endpoints, serializer=serializer)
    server._executor_cls = MagicMockIgnoreArgs
    server._create_listener = MagicMockIgnoreArgs()
    server.dispatcher = MagicMockIgnoreArgs()
    server.start()
    listener = server.listener
    server.stop()
    server.wait()
    self.assertEqual(1, listener.cleanup.call_count)