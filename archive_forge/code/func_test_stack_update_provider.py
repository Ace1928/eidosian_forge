import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_provider(self):
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_template'})
    files = {'provider.template': json.dumps(template)}
    env = {'resource_registry': {'My::TestResource': 'provider.template'}}
    stack_identifier = self.stack_create(template=self.provider_template, files=files, environment=env)
    initial_resources = {'test1': 'My::TestResource'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test1')
    nested_id = nested_identifier.split('/')[-1]
    nested_resources = {'test1': 'OS::Heat::TestResource'}
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))
    tmpl_update = _change_rsrc_properties(test_template_two_resource, ['test1', 'test2'], {'value': 'test_provider_template'})
    files['provider.template'] = json.dumps(tmpl_update)
    self.update_stack(stack_identifier, self.provider_template, environment=env, files=files)
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    rsrc = self.client.resources.get(stack_identifier, 'test1')
    self.assertEqual(rsrc.physical_resource_id, nested_id)
    nested_resources = {'test1': 'OS::Heat::TestResource', 'test2': 'OS::Heat::TestResource'}
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))