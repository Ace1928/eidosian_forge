from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
def test_stack_index(self):
    stacks = [self.stack1]
    stack_view = stacks_view.collection(self.request, stacks)
    self.assertIn('stacks', stack_view)
    self.assertEqual(1, len(stack_view['stacks']))