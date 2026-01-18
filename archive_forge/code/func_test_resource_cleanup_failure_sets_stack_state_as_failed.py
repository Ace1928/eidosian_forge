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
@mock.patch.object(stack.Stack, 'rollback')
def test_resource_cleanup_failure_sets_stack_state_as_failed(self, mock_tr, mock_cru, mock_crc, mock_pcr, mock_csc):
    self.is_update = False
    self.stack.state_set(self.stack.UPDATE, self.stack.IN_PROGRESS, '')
    self.resource.state_set(self.resource.UPDATE, self.resource.IN_PROGRESS)
    dummy_ex = exception.ResourceNotAvailable(resource_name=self.resource.name)
    mock_crc.side_effect = exception.ResourceFailure(dummy_ex, self.resource, action=self.resource.UPDATE)
    self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
    s = self.stack.load(self.ctx, stack_id=self.stack.id)
    self.assertEqual((s.UPDATE, s.FAILED), (s.action, s.status))
    self.assertEqual('Resource UPDATE failed: ResourceNotAvailable: resources.A: The Resource (A) is not available.', s.status_reason)