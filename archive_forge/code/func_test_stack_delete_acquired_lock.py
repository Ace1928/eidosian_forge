from unittest import mock
from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.rpc import dispatcher
from oslo_utils import timeutils
from heat.common import exception
from heat.common import service_utils
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.objects import stack as stack_object
from heat.objects import stack_lock as stack_lock_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(parser.Stack, 'load')
@mock.patch.object(stack_lock.StackLock, 'try_acquire')
def test_stack_delete_acquired_lock(self, mock_acquire, mock_load):
    mock_acquire.return_value = self.man.engine_id
    stack_name = 'service_delete_test_stack_acquired_lock'
    stack = tools.get_stack(stack_name, self.ctx)
    sid = stack.store()
    mock_load.return_value = stack
    st = stack_object.Stack.get_by_id(self.ctx, sid)
    self.assertIsNone(self.man.delete_stack(self.ctx, stack.identifier()))
    self.man.thread_group_mgr.groups[sid].wait()
    mock_acquire.assert_called_once_with()
    mock_load.assert_called_once_with(self.ctx, stack=st)