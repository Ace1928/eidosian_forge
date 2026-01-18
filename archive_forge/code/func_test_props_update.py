import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_props_update(self):
    """Test update of resource_def properties behaves as expected."""
    env = {'resource_registry': {'My::RandomString': 'OS::Heat::RandomString'}}
    template_one = self.template.replace('count: 0', 'count: 1')
    stack_identifier = self.stack_create(template=template_one, environment=env)
    self.assertEqual({u'random_group': u'OS::Heat::ResourceGroup'}, self.list_resources(stack_identifier))
    initial_nested_ident = self.group_nested_identifier(stack_identifier, 'random_group')
    self.assertEqual({'0': 'My::RandomString'}, self.list_resources(initial_nested_ident))
    res = self.client.resources.get(initial_nested_ident, '0')
    initial_res_id = res.physical_resource_id
    template_salt = template_one.replace('salt: initial', 'salt: more')
    self.update_stack(stack_identifier, template_salt, environment=env)
    updated_nested_ident = self.group_nested_identifier(stack_identifier, 'random_group')
    self.assertEqual(initial_nested_ident, updated_nested_ident)
    res = self.client.resources.get(updated_nested_ident, '0')
    updated_res_id = res.physical_resource_id
    self.assertNotEqual(initial_res_id, updated_res_id)