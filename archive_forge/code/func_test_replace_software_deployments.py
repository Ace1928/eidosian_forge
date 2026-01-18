import yaml
from heat_integrationtests.functional import functional_base
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