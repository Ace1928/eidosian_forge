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
class SteppingFakeExecutor(self.server._executor_cls):

    def __init__(self, *args, **kwargs):
        _runner[0] = eventlet.getcurrent()
        running_event.set()
        start_event.wait()
        super(SteppingFakeExecutor, self).__init__(*args, **kwargs)
        done_event.set()
        finish_event.wait()