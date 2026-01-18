import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_update_nochange(self):
    """Test update with no properties change."""
    env = {'resource_registry': {'My::RandomString': 'OS::Heat::RandomString'}}
    template_one = self.template.replace('count: 0', 'count: 2')
    stack_identifier = self.stack_create(template=template_one, environment=env)
    self.assertEqual({u'random_group': u'OS::Heat::ResourceGroup'}, self.list_resources(stack_identifier))
    initial_nested_ident = self.group_nested_identifier(stack_identifier, 'random_group')
    self.assertEqual({'0': 'My::RandomString', '1': 'My::RandomString'}, self.list_resources(initial_nested_ident))
    stack0 = self.client.stacks.get(stack_identifier)
    initial_rand = self._stack_output(stack0, 'random1')
    template_copy = copy.deepcopy(template_one)
    self.update_stack(stack_identifier, template_copy, environment=env)
    updated_nested_ident = self.group_nested_identifier(stack_identifier, 'random_group')
    self.assertEqual(initial_nested_ident, updated_nested_ident)
    stack1 = self.client.stacks.get(stack_identifier)
    updated_rand = self._stack_output(stack1, 'random1')
    self.assertEqual(initial_rand, updated_rand)