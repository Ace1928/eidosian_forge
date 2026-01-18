from unittest import mock
from heat.db import api as db_api
from heat.engine import check_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.engine import worker
from heat.objects import stack as stack_objects
from heat.rpc import worker_client as wc
from heat.tests import common
from heat.tests import utils
@mock.patch.object(worker, '_cancel_workers')
@mock.patch.object(worker.WorkerService, 'stop_traversal')
def test_stop_all_workers_when_stack_not_in_progress(self, mock_st, mock_cw):
    mock_tgm = mock.Mock()
    _worker = worker.WorkerService('host-1', 'topic-1', 'engine-001', mock_tgm)
    stack = mock.MagicMock()
    stack.FAILED = 'FAILED'
    stack.status = stack.FAILED
    stack.id = 'stack_id'
    stack.rollback = mock.MagicMock()
    _worker.stop_all_workers(stack)
    self.assertFalse(mock_st.called)
    mock_cw.assert_called_once_with(stack, mock_tgm, 'engine-001', _worker._rpc_client)
    self.assertFalse(stack.rollback.called)
    stack.FAILED = 'FAILED'
    stack.status = stack.FAILED
    _worker.stop_all_workers(stack)
    self.assertFalse(mock_st.called)
    mock_cw.assert_called_with(stack, mock_tgm, 'engine-001', _worker._rpc_client)
    self.assertFalse(stack.rollback.called)