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
class MagicMockIgnoreArgs(mock.MagicMock):
    """MagicMock ignores arguments.

            A MagicMock which can never misinterpret the arguments passed to
            it during construction.
            """

    def __init__(self, *args, **kwargs):
        super(MagicMockIgnoreArgs, self).__init__()