from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.engine import check_resource
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import sync_point
from heat.engine import worker
from heat.rpc import api as rpc_api
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_update_retrigger_check_resource_new_traversal_deletes_rsrc(self, mock_cru, mock_crc, mock_pcr, mock_csc):
    self.stack._convg_deps = dependencies.Dependencies([[(1, False), (1, True)], [(2, False), None]])
    self.cr.retrigger_check_resource(self.ctx, 2, self.stack)
    mock_pcr.assert_called_once_with(self.ctx, mock.ANY, 2, self.stack.current_traversal, mock.ANY, (2, False), None, False, None)