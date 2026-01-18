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
def test_resource_not_available(self, mock_cru, mock_crc, mock_pcr, mock_csc):
    self.worker.check_resource(self.ctx, 'non-existant-id', self.stack.current_traversal, {}, True, None)
    for mocked in [mock_cru, mock_crc, mock_pcr, mock_csc]:
        self.assertFalse(mocked.called)