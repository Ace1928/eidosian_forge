from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
class AzureRMDtlPolicy(AzureRMModuleBase):
    """Configuration class for an Azure RM Policy resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), policy_set_name=dict(type='str', required=True), name=dict(type='str', required=True), description=dict(type='str'), fact_name=dict(type='str', choices=['user_owned_lab_vm_count', 'user_owned_lab_premium_vm_count', 'lab_vm_count', 'lab_premium_vm_count', 'lab_vm_size', 'gallery_image', 'user_owned_lab_vm_count_in_subnet', 'lab_target_cost']), threshold=dict(type='raw'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.lab_name = None
        self.policy_set_name = None
        self.name = None
        self.policy = dict()
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        required_if = [('state', 'present', ['threshold', 'fact_name'])]
        super(AzureRMDtlPolicy, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True, required_if=required_if)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.policy[key] = kwargs[key]
        if self.state == 'present':
            self.policy['status'] = 'Enabled'
            dict_camelize(self.policy, ['fact_name'], True)
            if isinstance(self.policy['threshold'], list):
                self.policy['evaluator_type'] = 'AllowedValuesPolicy'
            else:
                self.policy['evaluator_type'] = 'MaxValuePolicy'
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        old_response = self.get_policy()
        if not old_response:
            self.log("Policy instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Policy instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                if not default_compare(self.policy, old_response, '', self.results):
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Policy instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_policy()
            self.results['changed'] = True
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Policy instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_policy()
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        else:
            self.log('Policy instance unchanged')
            self.results['changed'] = False
            response = old_response
        if self.state == 'present':
            self.results.update({'id': response.get('id', None), 'status': response.get('status', None)})
        return self.results

    def create_update_policy(self):
        """
        Creates or updates Policy with the specified configuration.

        :return: deserialized Policy instance state dictionary
        """
        self.log('Creating / Updating the Policy instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.policies.create_or_update(resource_group_name=self.resource_group, lab_name=self.lab_name, policy_set_name=self.policy_set_name, name=self.name, policy=self.policy)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the Policy instance.')
            self.fail('Error creating the Policy instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_policy(self):
        """
        Deletes specified Policy instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Policy instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.policies.delete(resource_group_name=self.resource_group, lab_name=self.lab_name, policy_set_name=self.policy_set_name, name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the Policy instance.')
            self.fail('Error deleting the Policy instance: {0}'.format(str(e)))
        return True

    def get_policy(self):
        """
        Gets the properties of the specified Policy.

        :return: deserialized Policy instance state dictionary
        """
        self.log('Checking if the Policy instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mgmt_client.policies.get(resource_group_name=self.resource_group, lab_name=self.lab_name, policy_set_name=self.policy_set_name, name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Policy instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Policy instance.')
        if found is True:
            return response.as_dict()
        return False