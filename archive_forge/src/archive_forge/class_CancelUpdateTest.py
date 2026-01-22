from heat_integrationtests.functional import functional_base
class CancelUpdateTest(functional_base.FunctionalTestsBase):
    template = "\nheat_template_version: '2013-05-23'\nparameters:\n InstanceType:\n   type: string\n ImageId:\n   type: string\n network:\n   type: string\nresources:\n port:\n   type: OS::Neutron::Port\n   properties:\n     network: {get_param: network}\n Server:\n   type: OS::Nova::Server\n   properties:\n     flavor_update_policy: REPLACE\n     image: {get_param: ImageId}\n     flavor: {get_param: InstanceType}\n     networks:\n       - port: {get_resource: port}\n"

    def setUp(self):
        super(CancelUpdateTest, self).setUp()
        if not self.conf.minimal_image_ref:
            raise self.skipException('No minimal image configured to test')
        if not self.conf.minimal_instance_type:
            raise self.skipException('No minimal flavor configured to test.')

    def test_cancel_update_server_with_port(self):
        parameters = {'InstanceType': self.conf.minimal_instance_type, 'ImageId': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name}
        stack_identifier = self.stack_create(template=self.template, parameters=parameters)
        parameters['InstanceType'] = self.conf.instance_type
        self.update_stack(stack_identifier, self.template, parameters=parameters, expected_status='UPDATE_IN_PROGRESS')
        self._wait_for_resource_status(stack_identifier, 'Server', 'CREATE_IN_PROGRESS')
        self.cancel_update_stack(stack_identifier)