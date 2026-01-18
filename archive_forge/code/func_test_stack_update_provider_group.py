import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_provider_group(self):
    """Test two-level nested update."""
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_group_template'})
    files = {'provider.template': json.dumps(template)}
    env = {'resource_registry': {'My::TestResource': 'provider.template'}}
    stack_identifier = self.stack_create(template=self.provider_group_template, files=files, environment=env)
    initial_resources = {'test_group': 'OS::Heat::ResourceGroup'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test_group')
    nested_resources = {'0': 'My::TestResource', '1': 'My::TestResource'}
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))
    for n_rsrc in nested_resources:
        rsrc = self.client.resources.get(nested_identifier, n_rsrc)
        provider_stack = self.client.stacks.get(rsrc.physical_resource_id)
        provider_identifier = '%s/%s' % (provider_stack.stack_name, provider_stack.id)
        provider_resources = {u'test1': u'OS::Heat::TestResource'}
        self.assertEqual(provider_resources, self.list_resources(provider_identifier))
    tmpl_update = _change_rsrc_properties(test_template_two_resource, ['test1', 'test2'], {'value': 'test_provider_group_template'})
    files['provider.template'] = json.dumps(tmpl_update)
    self.update_stack(stack_identifier, self.provider_group_template, environment=env, files=files)
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    nested_stack = self.client.stacks.get(nested_identifier)
    self.assertEqual('UPDATE_COMPLETE', nested_stack.stack_status)
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))
    for n_rsrc in nested_resources:
        rsrc = self.client.resources.get(nested_identifier, n_rsrc)
        provider_stack = self.client.stacks.get(rsrc.physical_resource_id)
        provider_identifier = '%s/%s' % (provider_stack.stack_name, provider_stack.id)
        provider_resources = {'test1': 'OS::Heat::TestResource', 'test2': 'OS::Heat::TestResource'}
        self.assertEqual(provider_resources, self.list_resources(provider_identifier))