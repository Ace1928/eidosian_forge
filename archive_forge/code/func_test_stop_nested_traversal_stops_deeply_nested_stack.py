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
@mock.patch.object(worker, '_stop_traversal')
def test_stop_nested_traversal_stops_deeply_nested_stack(self, mock_st):
    mock_tgm = mock.Mock()
    ctx = utils.dummy_context()
    tmpl = templatem.Template.create_empty_template()
    stack1 = parser.Stack(ctx, 'stack1', tmpl, current_traversal='123')
    stack1.store()
    stack2 = parser.Stack(ctx, 'stack2', tmpl, owner_id=stack1.id, current_traversal='456')
    stack2.store()
    stack3 = parser.Stack(ctx, 'stack3', tmpl, owner_id=stack2.id, current_traversal='789')
    stack3.store()
    _worker = worker.WorkerService('host-1', 'topic-1', 'engine-001', mock_tgm)
    _worker.stop_traversal(stack2)
    self.assertEqual(2, mock_st.call_count)
    call1, call2 = mock_st.call_args_list
    call_args1, call_args2 = (call1[0][0], call2[0][0])
    self.assertEqual('stack2', call_args1.name)
    self.assertEqual('stack3', call_args2.name)