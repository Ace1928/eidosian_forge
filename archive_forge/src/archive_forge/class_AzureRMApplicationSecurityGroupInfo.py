from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMApplicationSecurityGroupInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.resource_group = None
        self.name = None
        self.tags = None
        self.results = dict(changed=False)
        super(AzureRMApplicationSecurityGroupInfo, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        is_old_facts = self.module._name == 'azure_rm_applicationsecuritygroup_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_applicationsecuritygroup_facts' module has been renamed to 'azure_rm_applicationsecuritygroup_info'", version=(2.9,))
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        if self.name:
            if self.resource_group:
                self.results['applicationsecuritygroups'] = self.get()
            else:
                self.fail('resource_group is required when filtering by name')
        elif self.resource_group:
            self.results['applicationsecuritygroups'] = self.list_by_resource_group()
        else:
            self.results['applicationsecuritygroups'] = self.list_all()
        return self.results

    def get(self):
        """
        Gets the properties of the specified Application Security Group.

        :return: deserialized Application Security Group instance state dictionary
        """
        self.log('Get the Application Security Group instance {0}'.format(self.name))
        results = []
        try:
            response = self.network_client.application_security_groups.get(resource_group_name=self.resource_group, application_security_group_name=self.name)
            self.log('Response : {0}'.format(response))
            if response and self.has_tags(response.tags, self.tags):
                results.append(applicationsecuritygroup_to_dict(response))
        except ResourceNotFoundError as e:
            self.fail('Did not find the Application Security Group instance.')
        return results

    def list_by_resource_group(self):
        """
        Lists the properties of Application Security Groups in specific resource group.

        :return: deserialized Application Security Group instance state dictionary
        """
        self.log('Get the Application Security Groups in resource group {0}'.format(self.resource_group))
        results = []
        try:
            response = list(self.network_client.application_security_groups.list(resource_group_name=self.resource_group))
            self.log('Response : {0}'.format(response))
            if response:
                for item in response:
                    if self.has_tags(item.tags, self.tags):
                        results.append(applicationsecuritygroup_to_dict(item))
        except ResourceNotFoundError as e:
            self.log('Did not find the Application Security Group instance.')
        return results

    def list_all(self):
        """
        Lists the properties of Application Security Groups in specific subscription.

        :return: deserialized Application Security Group instance state dictionary
        """
        self.log('Get the Application Security Groups in current subscription')
        results = []
        try:
            response = list(self.network_client.application_security_groups.list_all())
            self.log('Response : {0}'.format(response))
            if response:
                for item in response:
                    if self.has_tags(item.tags, self.tags):
                        results.append(applicationsecuritygroup_to_dict(item))
        except ResourceNotFoundError as e:
            self.log('Did not find the Application Security Group instance.')
        return results