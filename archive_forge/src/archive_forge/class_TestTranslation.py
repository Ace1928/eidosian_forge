from heat_integrationtests.functional import functional_base
class TestTranslation(functional_base.FunctionalTestsBase):

    def test_create_update_subnet_old_network(self):
        env = {'parameters': {'net_cidr': '11.11.11.0/24'}}
        stack_identifier = self.stack_create(template=template_subnet_old_network, environment=env)
        env = {'parameters': {'net_cidr': '11.11.12.0/24'}}
        self.update_stack(stack_identifier, template=template_subnet_old_network, environment=env)

    def test_create_update_translation_with_get_attr(self):
        env = {'parameters': {'net_cidr': '11.11.11.0/24'}}
        stack_identifier = self.stack_create(template=template_with_get_attr, environment=env)
        env = {'parameters': {'net_cidr': '11.11.12.0/24'}}
        self.update_stack(stack_identifier, template=template_with_get_attr, environment=env)

    def test_value_from_nested_stack(self):
        env = {'parameters': {'flavor': self.conf.minimal_instance_type, 'image': self.conf.minimal_image_ref, 'public_net': self.conf.fixed_network_name}}
        self.stack_create(template=template_value_from_nested_stack_main, environment=env, files={'network.yaml': template_value_from_nested_stack_network})