from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMVMExtension(AzureRMModuleBase):
    """Configuration class for an Azure RM VM Extension resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), virtual_machine_name=dict(type='str', required=True), publisher=dict(type='str'), virtual_machine_extension_type=dict(type='str'), type_handler_version=dict(type='str'), auto_upgrade_minor_version=dict(type='bool'), settings=dict(type='dict'), protected_settings=dict(type='dict', no_log=True), force_update_tag=dict(type='bool', default=False))
        self.resource_group = None
        self.name = None
        self.virtual_machine_name = None
        self.location = None
        self.publisher = None
        self.virtual_machine_extension_type = None
        self.type_handler_version = None
        self.auto_upgrade_minor_version = None
        self.settings = None
        self.protected_settings = None
        self.state = None
        self.force_update_tag = False
        required_if = [('state', 'present', ['publisher', 'virtual_machine_extension_type', 'type_handler_version'])]
        self.results = dict(changed=False, state=dict())
        super(AzureRMVMExtension, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=False, supports_tags=False, required_if=required_if)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        if self.module._name == 'azure_rm_virtualmachine_extension':
            self.module.deprecate("The 'azure_rm_virtualmachine_extension' module has been renamed to 'azure_rm_virtualmachineextension'", version=(2, 9))
        resource_group = None
        to_be_updated = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        response = self.get_vmextension()
        if self.state == 'present':
            if not response:
                to_be_updated = True
            else:
                if self.force_update_tag:
                    to_be_updated = True
                if self.settings is not None:
                    if response['settings'] != self.settings:
                        response['settings'] = self.settings
                        to_be_updated = True
                else:
                    self.settings = response['settings']
                if response['location'] != self.location:
                    self.location = response['location']
                    self.module.warn("Property 'location' cannot be changed")
                if response['publisher'] != self.publisher:
                    self.publisher = response['publisher']
                    self.module.warn("Property 'publisher' cannot be changed")
                if response['virtual_machine_extension_type'] != self.virtual_machine_extension_type:
                    self.virtual_machine_extension_type = response['virtual_machine_extension_type']
                    self.module.warn("Property 'virtual_machine_extension_type' cannot be changed")
                if response['type_handler_version'] != self.type_handler_version:
                    response['type_handler_version'] = self.type_handler_version
                    to_be_updated = True
                if self.auto_upgrade_minor_version is not None:
                    if response['auto_upgrade_minor_version'] != self.auto_upgrade_minor_version:
                        response['auto_upgrade_minor_version'] = self.auto_upgrade_minor_version
                        to_be_updated = True
                else:
                    self.auto_upgrade_minor_version = response['auto_upgrade_minor_version']
            if to_be_updated:
                self.results['changed'] = True
                self.results['state'] = self.create_or_update_vmextension()
        elif self.state == 'absent':
            if response:
                self.delete_vmextension()
                self.results['changed'] = True
        return self.results

    def create_or_update_vmextension(self):
        """
        Method calling the Azure SDK to create or update the VM extension.
        :return: void
        """
        self.log('Creating VM extension {0}'.format(self.name))
        try:
            params = self.compute_models.VirtualMachineExtension(location=self.location, publisher=self.publisher, type_properties_type=self.virtual_machine_extension_type, type_handler_version=self.type_handler_version, auto_upgrade_minor_version=self.auto_upgrade_minor_version, settings=self.settings, protected_settings=self.protected_settings, force_update_tag=self.force_update_tag)
            poller = self.compute_client.virtual_machine_extensions.begin_create_or_update(self.resource_group, self.virtual_machine_name, self.name, params)
            response = self.get_poller_result(poller)
            return vmextension_to_dict(response)
        except Exception as e:
            self.log('Error attempting to create the VM extension.')
            self.fail('Error creating the VM extension: {0}'.format(str(e)))

    def delete_vmextension(self):
        """
        Method calling the Azure SDK to delete the VM Extension.
        :return: void
        """
        self.log('Deleting vmextension {0}'.format(self.name))
        try:
            poller = self.compute_client.virtual_machine_extensions.begin_delete(self.resource_group, self.virtual_machine_name, self.name)
            self.get_poller_result(poller)
        except Exception as e:
            self.log('Error attempting to delete the vmextension.')
            self.fail('Error deleting the vmextension: {0}'.format(str(e)))

    def get_vmextension(self):
        """
        Method calling the Azure SDK to get a VM Extension.
        :return: void
        """
        self.log('Checking if the vm extension {0} is present'.format(self.name))
        found = False
        try:
            response = self.compute_client.virtual_machine_extensions.get(self.resource_group, self.virtual_machine_name, self.name)
            found = True
        except ResourceNotFoundError as e:
            self.log('Did not find vm extension')
        if found:
            return vmextension_to_dict(response)
        else:
            return False