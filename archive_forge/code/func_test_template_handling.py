from oslo_utils import reflection
import heat.api.openstack.v1 as api_v1
from heat.tests import common
def test_template_handling(self):
    self.assertRoute(self.m, '/aaaa/resource_types', 'GET', 'list_resource_types', 'StackController', {'tenant_id': 'aaaa'})
    self.assertRoute(self.m, '/aaaa/resource_types/test_type', 'GET', 'resource_schema', 'StackController', {'tenant_id': 'aaaa', 'type_name': 'test_type'})
    self.assertRoute(self.m, '/aaaa/resource_types/test_type/template', 'GET', 'generate_template', 'StackController', {'tenant_id': 'aaaa', 'type_name': 'test_type'})
    self.assertRoute(self.m, '/aaaa/validate', 'POST', 'validate_template', 'StackController', {'tenant_id': 'aaaa'})