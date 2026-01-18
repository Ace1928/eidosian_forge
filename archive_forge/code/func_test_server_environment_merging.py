from heat_integrationtests.functional import functional_base
def test_server_environment_merging(self):
    files = {'env1.yaml': ENV_1, 'env2.yaml': ENV_2}
    environment_files = ['env1.yaml', 'env2.yaml']
    stack_id = self.stack_create(stack_name='env_merge', template=TEMPLATE, files=files, environment_files=environment_files)
    resources = self.list_resources(stack_id)
    self.assertEqual(4, len(resources))
    stack = self.client.stacks.get(stack_id)
    self.assertEqual('CORRECT', stack.parameters['p0'])
    self.assertEqual('CORRECT', stack.parameters['p1'])
    self.assertEqual('CORRECT', stack.parameters['p2'])
    r3b = self.client.resources.get(stack_id, 'r3b')
    r3b_attrs = r3b.attributes
    self.assertIn('value', r3b_attrs)