import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_resource_group_zero_novalidate(self):
    nested_template_fail = '\nheat_template_version: 2013-05-23\nparameters:\n  length:\n    type: string\n    default: 50\n  salt:\n    type: string\n    default: initial\nresources:\n  random:\n    type: OS::Heat::RandomString\n    properties:\n      length: BAD\n'
    files = {'provider.yaml': nested_template_fail}
    env = {'resource_registry': {'My::RandomString': 'provider.yaml'}}
    stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
    self.assertEqual({u'random_group': u'OS::Heat::ResourceGroup'}, self.list_resources(stack_identifier))
    nested_identifier = self.group_nested_identifier(stack_identifier, 'random_group')
    self.assertEqual({}, self.list_resources(nested_identifier))
    template_two_nested = self.template.replace('count: 0', 'count: 2')
    expected_err = "resources.random_group<nested_stack>.resources.0<provider.yaml>.resources.random: : Value 'BAD' is not an integer"
    ex = self.assertRaises(exc.HTTPBadRequest, self.update_stack, stack_identifier, template_two_nested, environment=env, files=files)
    self.assertIn(expected_err, str(ex))
    ex = self.assertRaises(exc.HTTPBadRequest, self.stack_create, template=template_two_nested, environment=env, files=files)
    self.assertIn(expected_err, str(ex))