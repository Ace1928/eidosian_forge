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
def test_resource_cleanup_failure_triggers_rollback_if_enabled(self, mock_cru, mock_crc, mock_pcr, mock_csc):
    mock_tr = self.stack.rollback = mock.Mock(return_value=None)
    self.is_update = False
    self.stack.disable_rollback = False
    self.stack.store()
    dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
    mock_crc.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
    mock_tr.assert_called_once_with()