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
@mock.patch.object(resource.Resource, 'state_set')
def test_try_steal_lock_alive(self, mock_ss, mock_cru, mock_crc, mock_pcr, mock_csc):
    res = self.cr._stale_resource_needs_retry(self.ctx, self.resource, str(uuid.uuid4()))
    self.assertFalse(res)
    mock_ss.assert_not_called()