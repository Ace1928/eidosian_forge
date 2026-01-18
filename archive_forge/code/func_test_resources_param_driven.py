from heat_integrationtests.functional import functional_base
def test_resources_param_driven(self):
    params = {'chain-types': 'OS::Heat::None,OS::Heat::RandomString,OS::Heat::None'}
    stack_id = self.stack_create(template=TEMPLATE_PARAM_DRIVEN, parameters=params)
    nested_id = self.group_nested_identifier(stack_id, 'my-chain')
    expected = {'0': 'OS::Heat::None', '1': 'OS::Heat::RandomString', '2': 'OS::Heat::None'}
    found = self.list_resources(nested_id)
    self.assertEqual(expected, found)