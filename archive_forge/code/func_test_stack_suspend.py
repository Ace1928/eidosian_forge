from unittest import mock
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(stack.Stack, 'load')
@mock.patch.object(service.ThreadGroupManager, 'start')
def test_stack_suspend(self, mock_start, mock_load):
    stack_name = 'service_suspend_test_stack'
    t = template_format.parse(tools.wp_template)
    stk = utils.parse_stack(t, stack_name=stack_name)
    s = stack_object.Stack.get_by_id(self.ctx, stk.id)
    mock_load.return_value = stk
    thread = mock.MagicMock()
    mock_link = self.patchobject(thread, 'link')
    mock_start.return_value = thread
    self.patchobject(service, 'NotifyEvent')
    result = self.man.stack_suspend(self.ctx, stk.identifier())
    self.assertIsNone(result)
    mock_load.assert_called_once_with(self.ctx, stack=s)
    mock_link.assert_called_once_with(mock.ANY)
    mock_start.assert_called_once_with(stk.id, stk.suspend, notify=mock.ANY)
    stk.delete()