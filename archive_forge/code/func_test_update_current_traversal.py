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
@mock.patch.object(stack_objects.Stack, 'select_and_update')
def test_update_current_traversal(self, mock_sau):
    stack = mock.MagicMock()
    stack.current_traversal = 'some-thing'
    old_trvsl = stack.current_traversal
    worker._update_current_traversal(stack)
    self.assertNotEqual(old_trvsl, stack.current_traversal)
    mock_sau.assert_called_once_with(mock.ANY, stack.id, mock.ANY, exp_trvsl=old_trvsl)