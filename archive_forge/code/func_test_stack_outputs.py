from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_stack_outputs(self):
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/outputs', 'GET', 'list_outputs', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/outputs/cccc', 'GET', 'show_output', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb', 'output_key': 'cccc'})