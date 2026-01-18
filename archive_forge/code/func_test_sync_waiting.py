from unittest import mock
from oslo_db import exception
from heat.engine import sync_point
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def test_sync_waiting(self):
    ctx = utils.dummy_context()
    stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
    stack.converge_stack(stack.t, action=stack.CREATE)
    resource = stack['C']
    graph = stack.convergence_dependencies.graph()
    sender = (4, True)
    mock_callback = mock.Mock()
    sync_point.sync(ctx, resource.id, stack.current_traversal, True, mock_callback, set(graph[resource.id, True]), {sender: None})
    updated_sync_point = sync_point.get(ctx, resource.id, stack.current_traversal, True)
    input_data = sync_point.deserialize_input_data(updated_sync_point.input_data)
    self.assertEqual({sender: None}, input_data)
    self.assertFalse(mock_callback.called)