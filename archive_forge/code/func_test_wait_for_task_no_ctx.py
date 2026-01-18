from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch.object(context, 'get_current', return_value=None)
def test_wait_for_task_no_ctx(self, mock_curr_ctx):
    api_session = self._create_api_session(True)
    task_info_list = [('queued', 0), ('running', 40), ('success', 100)]
    task_info_list_size = len(task_info_list)

    def invoke_api_side_effect(module, method, *args, **kwargs):
        state, progress = task_info_list.pop(0)
        task_info = mock.Mock()
        task_info.progress = progress
        task_info.queueTime = datetime(2016, 12, 6, 15, 29, 43, 79060)
        task_info.completeTime = datetime(2016, 12, 6, 15, 29, 50, 79060)
        task_info.state = state
        return task_info
    api_session.invoke_api = mock.Mock(side_effect=invoke_api_side_effect)
    task = mock.Mock()
    with mock.patch.object(greenthread, 'sleep'):
        ret = api_session.wait_for_task(task)
        self.assertEqual('success', ret.state)
        self.assertEqual(100, ret.progress)
    api_session.invoke_api.assert_called_with(vim_util, 'get_object_property', api_session.vim, task, 'info', skip_op_id=True)
    self.assertEqual(task_info_list_size, api_session.invoke_api.call_count)
    mock_curr_ctx.assert_called_once()