from heat_integrationtests.functional import functional_base
class StackValidationTest(functional_base.FunctionalTestsBase):

    def setUp(self):
        super(StackValidationTest, self).setUp()
        if not self.conf.minimal_image_ref:
            raise self.skipException('No image configured to test')
        if not self.conf.minimal_instance_type:
            raise self.skipException('No minimal_instance_type configured to test')
        self.assign_keypair()

    def test_stack_validate_provider_references_parent_resource(self):
        template = '\nheat_template_version: 2014-10-16\nparameters:\n  keyname:\n    type: string\n  flavor:\n    type: string\n  image:\n    type: string\n  network:\n    type: string\nresources:\n  config:\n    type: My::Config\n    properties:\n        server: {get_resource: server}\n\n  server:\n    type: OS::Nova::Server\n    properties:\n      image: {get_param: image}\n      flavor: {get_param: flavor}\n      key_name: {get_param: keyname}\n      networks: [{network: {get_param: network} }]\n      user_data_format: SOFTWARE_CONFIG\n\n'
        config_template = '\nheat_template_version: 2014-10-16\nparameters:\n  server:\n    type: string\nresources:\n  config:\n    type: OS::Heat::SoftwareConfig\n\n  deployment:\n    type: OS::Heat::SoftwareDeployment\n    properties:\n      config:\n        get_resource: config\n      server:\n        get_param: server\n'
        files = {'provider.yaml': config_template}
        env = {'resource_registry': {'My::Config': 'provider.yaml'}}
        parameters = {'keyname': self.keypair_name, 'flavor': self.conf.minimal_instance_type, 'image': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name}
        self.stack_create(template=template, files=files, environment=env, parameters=parameters, expected_status='CREATE_IN_PROGRESS')