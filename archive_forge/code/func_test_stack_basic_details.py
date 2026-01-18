from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
@mock.patch.object(stacks_view, 'format_stack')
def test_stack_basic_details(self, mock_format_stack):
    stacks = [self.stack1]
    expected_keys = stacks_view.basic_keys
    stacks_view.collection(self.request, stacks)
    mock_format_stack.assert_called_once_with(self.request, self.stack1, expected_keys, mock.ANY)