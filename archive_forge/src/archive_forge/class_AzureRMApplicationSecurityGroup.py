from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMApplicationSecurityGroup(AzureRMModuleBase):
    """Configuration class for an Azure RM Application Security Group resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.location = None
        self.name = None
        self.tags = None
        self.state = None
        self.results = dict(changed=False)
        self.to_do = Actions.NoAction
        super(AzureRMApplicationSecurityGroup, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        old_response = self.get_applicationsecuritygroup()
        if not old_response:
            self.log("Application Security Group instance doesn't exist")
            if self.state == 'present':
                self.to_do = Actions.CreateOrUpdate
            else:
                self.log("Old instance didn't exist")
        else:
            self.log('Application Security Group instance already exists')
            if self.state == 'present':
                if self.check_update(old_response):
                    self.to_do = Actions.CreateOrUpdate
                update_tags, self.tags = self.update_tags(old_response.get('tags', None))
                if update_tags:
                    self.to_do = Actions.CreateOrUpdate
            elif self.state == 'absent':
                self.to_do = Actions.Delete
        if self.to_do == Actions.CreateOrUpdate:
            self.log('Need to Create / Update the Application Security Group instance')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_applicationsecuritygroup()
            self.results['id'] = response['id']
        elif self.to_do == Actions.Delete:
            self.log('Delete Application Security Group instance')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_applicationsecuritygroup()
        return self.results

    def check_update(self, existing_asg):
        if self.location and self.location.lower() != existing_asg['location'].lower():
            self.module.warn('location cannot be updated. Existing {0}, input {1}'.format(existing_asg['location'], self.location))
        return False

    def create_update_applicationsecuritygroup(self):
        """
        Create or update Application Security Group.

        :return: deserialized Application Security Group instance state dictionary
        """
        self.log('Creating / Updating the Application Security Group instance {0}'.format(self.name))
        param = dict(name=self.name, tags=self.tags, location=self.location)
        try:
            response = self.network_client.application_security_groups.begin_create_or_update(resource_group_name=self.resource_group, application_security_group_name=self.name, parameters=param)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error creating/updating Application Security Group instance.')
            self.fail('Error creating/updating Application Security Group instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_applicationsecuritygroup(self):
        """
        Deletes specified Application Security Group instance.

        :return: True
        """
        self.log('Deleting the Application Security Group instance {0}'.format(self.name))
        try:
            response = self.network_client.application_security_groups.begin_delete(resource_group_name=self.resource_group, application_security_group_name=self.name)
        except Exception as e:
            self.log('Error deleting the Application Security Group instance.')
            self.fail('Error deleting the Application Security Group instance: {0}'.format(str(e)))
        return True

    def get_applicationsecuritygroup(self):
        """
        Gets the properties of the specified Application Security Group.

        :return: deserialized Application Security Group instance state dictionary
        """
        self.log('Checking if the Application Security Group instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.network_client.application_security_groups.get(resource_group_name=self.resource_group, application_security_group_name=self.name)
            self.log('Response : {0}'.format(response))
            self.log('Application Security Group instance : {0} found'.format(response.name))
            return response.as_dict()
        except ResourceNotFoundError as e:
            self.log('Did not find the Application Security Group instance.')
        return False