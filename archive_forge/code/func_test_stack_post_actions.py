from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_stack_post_actions(self):
    self.assertRoute(self.m, '/aaaa/stacks/teststack/bbbb/actions', 'POST', 'action', 'ActionController', {'tenant_id': 'aaaa', 'stack_name': 'teststack', 'stack_id': 'bbbb'})