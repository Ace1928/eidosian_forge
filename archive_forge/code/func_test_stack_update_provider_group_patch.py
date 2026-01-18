import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_provider_group_patch(self):
    """Test two-level nested update with PATCH"""
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_group_template'})
    files = {'provider.template': json.dumps(template)}
    env = {'resource_registry': {'My::TestResource': 'provider.template'}}
    stack_identifier = self.stack_create(template=self.provider_group_template, files=files, environment=env)
    initial_resources = {'test_group': 'OS::Heat::ResourceGroup'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test_group')
    nested_resources = {'0': 'My::TestResource', '1': 'My::TestResource'}
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))
    params = {'count': 3}
    self.update_stack(stack_identifier, parameters=params, existing=True)
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    nested_stack = self.client.stacks.get(nested_identifier)
    self.assertEqual('UPDATE_COMPLETE', nested_stack.stack_status)
    nested_resources['2'] = 'My::TestResource'
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))