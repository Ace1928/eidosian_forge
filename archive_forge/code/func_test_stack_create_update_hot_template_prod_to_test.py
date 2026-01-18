from heat_integrationtests.functional import functional_base
def test_stack_create_update_hot_template_prod_to_test(self):
    parms = {'env_type': 'prod'}
    stack_identifier = self.stack_create(template=hot_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier)
    self.output_assert_for_prod(stack_identifier)
    parms = {'env_type': 'prod', 'zone': 'xianyang'}
    self.update_stack(stack_identifier, template=hot_template, parameters=parms)
    self.res_assert_for_prod(stack_identifier, False, shannxi_provice=True)
    self.output_assert_for_prod(stack_identifier, False)
    parms = {'env_type': 'test'}
    self.update_stack(stack_identifier, template=hot_template, parameters=parms)
    self.res_assert_for_test(stack_identifier)
    self.output_assert_for_test(stack_identifier)
    parms = {'env_type': 'test', 'zone': 'xianyang'}
    self.update_stack(stack_identifier, template=hot_template, parameters=parms)
    self.res_assert_for_test(stack_identifier, fj_zone=False, shannxi_provice=True)
    self.output_assert_for_test(stack_identifier)