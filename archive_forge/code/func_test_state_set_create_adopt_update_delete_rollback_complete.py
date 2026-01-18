from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_state_set_create_adopt_update_delete_rollback_complete(self, mock_ps):
    mock_ps.return_value = 'updated'
    ret_val = self.stack.state_set(self.stack.CREATE, self.stack.COMPLETE, 'Create complete')
    self.assertTrue(mock_ps.called)
    self.assertEqual('updated', ret_val)
    mock_ps.reset_mock()
    ret_val = self.stack.state_set(self.stack.UPDATE, self.stack.COMPLETE, 'Update complete')
    self.assertTrue(mock_ps.called)
    self.assertEqual('updated', ret_val)
    mock_ps.reset_mock()
    ret_val = self.stack.state_set(self.stack.ROLLBACK, self.stack.COMPLETE, 'Rollback complete')
    self.assertTrue(mock_ps.called)
    self.assertEqual('updated', ret_val)
    mock_ps.reset_mock()
    ret_val = self.stack.state_set(self.stack.DELETE, self.stack.COMPLETE, 'Delete complete')
    self.assertTrue(mock_ps.called)
    self.assertEqual('updated', ret_val)
    mock_ps.reset_mock()
    ret_val = self.stack.state_set(self.stack.ADOPT, self.stack.COMPLETE, 'Adopt complete')
    self.assertTrue(mock_ps.called)
    self.assertEqual('updated', ret_val)