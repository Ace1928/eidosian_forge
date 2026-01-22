import os
import queue
import time
import uuid
import fixtures
from oslo_config import cfg
import oslo_messaging
from oslo_messaging._drivers.kafka_driver import kafka_options
from oslo_messaging.notify import notifier
from oslo_messaging.tests import utils as test_utils
class RpcServerFixture(fixtures.Fixture):
    """Fixture to setup the TestServerEndpoint."""

    def __init__(self, conf, url, target, endpoint=None, ctrl_target=None, executor='eventlet'):
        super(RpcServerFixture, self).__init__()
        self.conf = conf
        self.url = url
        self.target = target
        self.endpoint = endpoint or TestServerEndpoint()
        self.executor = executor
        self.syncq = queue.Queue()
        self.ctrl_target = ctrl_target or self.target

    def setUp(self):
        super(RpcServerFixture, self).setUp()
        endpoints = [self.endpoint, self]
        transport = self.useFixture(RPCTransportFixture(self.conf, self.url))
        self.server = oslo_messaging.get_rpc_server(transport=transport.transport, target=self.target, endpoints=endpoints, executor=self.executor)
        self._ctrl = oslo_messaging.get_rpc_client(transport.transport, self.ctrl_target)
        self._start()
        transport.wait()

    def cleanUp(self):
        self._stop()
        super(RpcServerFixture, self).cleanUp()

    def _start(self):
        self.thread = test_utils.ServerThreadHelper(self.server)
        self.thread.start()

    def _stop(self):
        self.thread.stop()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise Exception('Server did not shutdown correctly')

    def ping(self, ctxt):
        pass

    def sync(self, ctxt):
        self.syncq.put('x')