import datetime
from unittest import mock
from oslo_config import cfg
from oslo_utils import timeutils
from heat.common import context
from heat.common import service_utils
from heat.engine import service
from heat.engine import worker
from heat.objects import service as service_objects
from heat.rpc import worker_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_stop_rpc_server(self):
    with mock.patch.object(self.eng, '_rpc_server') as mock_rpc_server:
        self.eng._stop_rpc_server()
        mock_rpc_server.stop.assert_called_once_with()
        mock_rpc_server.wait.assert_called_once_with()