import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_provider_type(self):
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_template'})
    files = {'provider.template': json.dumps(template)}
    env = {'resource_registry': {'My::TestResource': 'provider.template', 'My::TestResource2': 'provider.template'}}
    stack_identifier = self.stack_create(template=self.provider_template, files=files, environment=env)
    p_res = self.client.resources.get(stack_identifier, 'test1')
    self.assertEqual('My::TestResource', p_res.resource_type)
    initial_resources = {'test1': 'My::TestResource'}
    self.assertEqual(initial_resources, self.list_resources(stack_identifier))
    nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test1')
    nested_id = nested_identifier.split('/')[-1]
    nested_resources = {'test1': 'OS::Heat::TestResource'}
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))
    n_res = self.client.resources.get(nested_identifier, 'test1')
    tmpl_update = copy.deepcopy(self.provider_template)
    tmpl_update['resources']['test1']['type'] = 'My::TestResource2'
    self.update_stack(stack_identifier, tmpl_update, environment=env, files=files)
    p_res = self.client.resources.get(stack_identifier, 'test1')
    self.assertEqual('My::TestResource2', p_res.resource_type)
    self.assertEqual({u'test1': u'My::TestResource2'}, self.list_resources(stack_identifier))
    rsrc = self.client.resources.get(stack_identifier, 'test1')
    self.assertEqual(rsrc.physical_resource_id, nested_id)
    self.assertEqual(nested_resources, self.list_resources(nested_identifier))
    n_res2 = self.client.resources.get(nested_identifier, 'test1')
    self.assertEqual(n_res.physical_resource_id, n_res2.physical_resource_id)