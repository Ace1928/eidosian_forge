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
class ServerSetupMixin(object):

    class Server(object):

        def __init__(self, transport, topic, server, endpoint, serializer, exchange):
            self.controller = ServerSetupMixin.ServerController()
            target = oslo_messaging.Target(topic=topic, server=server, exchange=exchange)
            self.server = oslo_messaging.get_rpc_server(transport, target, [endpoint, self.controller], serializer=serializer)

        def wait(self):
            self.controller.stopped.wait()
            self.server.start()
            self.server.stop()
            self.server.wait()

        def start(self):
            self.server.start()

    class ServerController(object):

        def __init__(self):
            self.stopped = eventletutils.Event()

        def stop(self, ctxt):
            self.stopped.set()

    class TestSerializer(object):

        def serialize_entity(self, ctxt, entity):
            return 's' + entity if entity else entity

        def deserialize_entity(self, ctxt, entity):
            return 'd' + entity if entity else entity

        def serialize_context(self, ctxt):
            return dict([(k, 's' + v) for k, v in ctxt.items()])

        def deserialize_context(self, ctxt):
            return dict([(k, 'd' + v) for k, v in ctxt.items()])

    def __init__(self):
        self.serializer = self.TestSerializer()

    def _setup_server(self, transport, endpoint, topic=None, server=None, exchange=None):
        server = self.Server(transport, topic=topic or 'testtopic', server=server or 'testserver', endpoint=endpoint, serializer=self.serializer, exchange=exchange)
        server.start()
        return server

    def _stop_server(self, client, server, topic=None, exchange=None):
        client.cast({}, 'stop')
        server.wait()

    def _setup_client(self, transport, topic='testtopic', exchange=None):
        target = oslo_messaging.Target(topic=topic, exchange=exchange)
        return oslo_messaging.get_rpc_client(transport, target=target, serializer=self.serializer)