import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_update_nochange_resource_needs_update(self):
    """Test update when the resource definition has changed.

        Test the scenario when the ResourceGroup update happens without
        any changed properties, this can happen if the definition of
        a contained provider resource changes (files map changes), then
        the group and underlying nested stack should end up updated.
        """
    random_templ1 = '\nheat_template_version: 2013-05-23\nparameters:\n  length:\n    type: string\n    default: not-used\n  salt:\n    type: string\n    default: not-used\nresources:\n  random1:\n    type: OS::Heat::RandomString\n    properties:\n      salt: initial\noutputs:\n  value:\n    value: {get_attr: [random1, value]}\n'
    files1 = {'my_random.yaml': random_templ1}
    random_templ2 = random_templ1.replace('salt: initial', 'salt: more')
    files2 = {'my_random.yaml': random_templ2}
    env = {'resource_registry': {'My::RandomString': 'my_random.yaml'}}
    template_one = self.template.replace('count: 0', 'count: 2')
    stack_identifier = self.stack_create(template=template_one, environment=env, files=files1)
    self.assertEqual({u'random_group': u'OS::Heat::ResourceGroup'}, self.list_resources(stack_identifier))
    self.assertEqual(files1, self.client.stacks.files(stack_identifier))
    initial_nested_ident = self.group_nested_identifier(stack_identifier, 'random_group')
    self.assertEqual({'0': 'My::RandomString', '1': 'My::RandomString'}, self.list_resources(initial_nested_ident))
    stack0 = self.client.stacks.get(stack_identifier)
    initial_rand = self._stack_output(stack0, 'random1')
    self.update_stack(stack_identifier, template_one, environment=env, files=files2)
    updated_nested_ident = self.group_nested_identifier(stack_identifier, 'random_group')
    self.assertEqual(initial_nested_ident, updated_nested_ident)
    self.assertEqual(files2, self.client.stacks.files(stack_identifier))
    stack1 = self.client.stacks.get(stack_identifier)
    updated_rand = self._stack_output(stack1, 'random1')
    self.assertNotEqual(initial_rand, updated_rand)