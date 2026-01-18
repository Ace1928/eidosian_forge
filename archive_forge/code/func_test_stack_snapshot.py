from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_stack_snapshot(self):
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/snapshots', 'POST', 'snapshot', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/snapshots/cccc', 'GET', 'show_snapshot', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb', 'snapshot_id': 'cccc'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/snapshots/cccc', 'DELETE', 'delete_snapshot', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb', 'snapshot_id': 'cccc'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/snapshots', 'GET', 'list_snapshots', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb'})
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/snapshots/cccc/restore', 'POST', 'restore_snapshot', 'StackController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb', 'snapshot_id': 'cccc'})