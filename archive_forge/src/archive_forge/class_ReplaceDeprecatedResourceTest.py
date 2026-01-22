import yaml
from heat_integrationtests.functional import functional_base
class ReplaceDeprecatedResourceTest(functional_base.FunctionalTestsBase):
    template = '\nheat_template_version: "2013-05-23"\nparameters:\n  flavor:\n    type: string\n  image:\n    type: string\n  network:\n    type: string\n\nresources:\n  config:\n    type: OS::Heat::SoftwareConfig\n    properties:\n      config: xxxx\n\n  server:\n    type: OS::Nova::Server\n    properties:\n      image: {get_param: image}\n      flavor: {get_param: flavor}\n      networks: [{network: {get_param: network} }]\n      user_data_format: SOFTWARE_CONFIG\n  dep:\n    type: OS::Heat::SoftwareDeployments\n    properties:\n        config: {get_resource: config}\n        servers: {\'0\': {get_resource: server}}\n        signal_transport: NO_SIGNAL\noutputs:\n  server:\n    value: {get_resource: server}\n'
    deployment_group_snippet = "\ntype: OS::Heat::SoftwareDeploymentGroup\nproperties:\n  config: {get_resource: config}\n  servers: {'0': {get_resource: server}}\n  signal_transport: NO_SIGNAL\n"
    enable_cleanup = True

    def test_replace_software_deployments(self):
        parms = {'flavor': self.conf.minimal_instance_type, 'network': self.conf.fixed_network_name, 'image': self.conf.minimal_image_ref}
        deployments_template = yaml.safe_load(self.template)
        stack_identifier = self.stack_create(parameters=parms, template=deployments_template, enable_cleanup=self.enable_cleanup)
        expected_resources = {'config': 'OS::Heat::SoftwareConfig', 'dep': 'OS::Heat::SoftwareDeployments', 'server': 'OS::Nova::Server'}
        self.assertEqual(expected_resources, self.list_resources(stack_identifier))
        resource = self.client.resources.get(stack_identifier, 'dep')
        initial_phy_id = resource.physical_resource_id
        resources = deployments_template['resources']
        resources['dep'] = yaml.safe_load(self.deployment_group_snippet)
        self.update_stack(stack_identifier, deployments_template, parameters=parms)
        expected_new_resources = {'config': 'OS::Heat::SoftwareConfig', 'dep': 'OS::Heat::SoftwareDeploymentGroup', 'server': 'OS::Nova::Server'}
        self.assertEqual(expected_new_resources, self.list_resources(stack_identifier))
        resource = self.client.resources.get(stack_identifier, 'dep')
        self.assertEqual(initial_phy_id, resource.physical_resource_id)