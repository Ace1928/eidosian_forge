from heat_integrationtests.functional import functional_base
def test_resources_env_defined(self):
    env = {'parameters': {'chain-types': 'OS::Heat::None'}}
    stack_id = self.stack_create(template=TEMPLATE_PARAM_DRIVEN, environment=env)
    nested_id = self.group_nested_identifier(stack_id, 'my-chain')
    expected = {'0': 'OS::Heat::None'}
    found = self.list_resources(nested_id)
    self.assertEqual(expected, found)