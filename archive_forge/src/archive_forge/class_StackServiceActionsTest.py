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
class StackServiceActionsTest(common.HeatTestCase):

    def setUp(self):
        super(StackServiceActionsTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.man = service.EngineService('a-host', 'a-topic')
        self.man.thread_group_mgr = service.ThreadGroupManager()

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

    @mock.patch.object(stack.Stack, 'load')
    @mock.patch.object(service.ThreadGroupManager, 'start')
    def test_stack_resume(self, mock_start, mock_load):
        stack_name = 'service_resume_test_stack'
        t = template_format.parse(tools.wp_template)
        stk = utils.parse_stack(t, stack_name=stack_name)
        mock_load.return_value = stk
        thread = mock.MagicMock()
        mock_link = self.patchobject(thread, 'link')
        mock_start.return_value = thread
        self.patchobject(service, 'NotifyEvent')
        result = self.man.stack_resume(self.ctx, stk.identifier())
        self.assertIsNone(result)
        mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)
        mock_link.assert_called_once_with(mock.ANY)
        mock_start.assert_called_once_with(stk.id, stk.resume, notify=mock.ANY)
        stk.delete()

    def test_stack_suspend_nonexist(self):
        stack_name = 'service_suspend_nonexist_test_stack'
        t = template_format.parse(tools.wp_template)
        tmpl = templatem.Template(t)
        stk = stack.Stack(self.ctx, stack_name, tmpl)
        ex = self.assertRaises(dispatcher.ExpectedException, self.man.stack_suspend, self.ctx, stk.identifier())
        self.assertEqual(exception.EntityNotFound, ex.exc_info[0])

    def test_stack_resume_nonexist(self):
        stack_name = 'service_resume_nonexist_test_stack'
        t = template_format.parse(tools.wp_template)
        tmpl = templatem.Template(t)
        stk = stack.Stack(self.ctx, stack_name, tmpl)
        ex = self.assertRaises(dispatcher.ExpectedException, self.man.stack_resume, self.ctx, stk.identifier())
        self.assertEqual(exception.EntityNotFound, ex.exc_info[0])

    def _mock_thread_start(self, stack_id, func, *args, **kwargs):
        func(*args, **kwargs)
        return mock.Mock()

    @mock.patch.object(service.ThreadGroupManager, 'start')
    @mock.patch.object(stack.Stack, 'load')
    def test_stack_check(self, mock_load, mock_start):
        stack_name = 'service_check_test_stack'
        t = template_format.parse(tools.wp_template)
        stk = utils.parse_stack(t, stack_name=stack_name)
        stk.check = mock.Mock()
        self.patchobject(service, 'NotifyEvent')
        mock_load.return_value = stk
        mock_start.side_effect = self._mock_thread_start
        self.man.stack_check(self.ctx, stk.identifier())
        self.assertTrue(stk.check.called)
        stk.delete()