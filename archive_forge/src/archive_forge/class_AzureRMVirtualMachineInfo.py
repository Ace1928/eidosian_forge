from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.six.moves.urllib.parse import urlparse
import re
class AzureRMVirtualMachineInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False, vms=[])
        self.resource_group = None
        self.name = None
        self.tags = None
        super(AzureRMVirtualMachineInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_virtualmachine_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_virtualmachine_facts' module has been renamed to 'azure_rm_virtualmachine_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        if self.name:
            self.results['vms'] = self.get_item()
        elif self.resource_group:
            self.results['vms'] = self.list_items_by_resourcegroup()
        else:
            self.results['vms'] = self.list_all_items()
        return self.results

    def get_item(self):
        self.log('Get properties for {0}'.format(self.name))
        item = None
        result = []
        item = self.get_vm(self.resource_group, self.name)
        if item and self.has_tags(item.get('tags'), self.tags):
            result = [item]
        return result

    def list_items_by_resourcegroup(self):
        self.log('List all items')
        try:
            items = self.compute_client.virtual_machines.list(self.resource_group)
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in items:
            if self.has_tags(item.tags, self.tags):
                results.append(self.get_vm(self.resource_group, item.name))
        return results

    def list_all_items(self):
        self.log('List all items')
        try:
            items = self.compute_client.virtual_machines.list_all()
        except ResourceNotFoundError as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        results = []
        for item in items:
            if self.has_tags(item.tags, self.tags):
                results.append(self.get_vm(parse_resource_id(item.id).get('resource_group'), item.name))
        return results

    def get_vm(self, resource_group, name):
        """
        Get the VM with expanded instanceView

        :return: VirtualMachine object
        """
        try:
            vm = self.compute_client.virtual_machines.get(resource_group, name, expand='instanceview')
            return self.serialize_vm(vm)
        except ResourceNotFoundError as exc:
            self.fail('Error getting virtual machine {0} - {1}'.format(self.name, str(exc)))

    def serialize_vm(self, vm):
        """
        Convert a VirtualMachine object to dict.

        :param vm: VirtualMachine object
        :return: dict
        """
        result = self.serialize_obj(vm, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES)
        resource_group = parse_resource_id(result['id']).get('resource_group')
        instance = None
        power_state = None
        display_status = None
        try:
            instance = self.compute_client.virtual_machines.instance_view(resource_group, vm.name)
            instance = self.serialize_obj(instance, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES)
        except Exception as exc:
            self.fail('Error getting virtual machine {0} instance view - {1}'.format(vm.name, str(exc)))
        for index in range(len(instance['statuses'])):
            code = instance['statuses'][index]['code'].split('/')
            if code[0] == 'PowerState':
                power_state = code[1]
                display_status = instance['statuses'][index]['display_status']
            elif code[0] == 'OSState' and code[1] == 'generalized':
                display_status = instance['statuses'][index]['display_status']
                power_state = 'generalized'
                break
            elif code[0] == 'ProvisioningState' and code[1] == 'failed':
                display_status = instance['statuses'][index]['display_status']
                power_state = ''
                break
        new_result = {}
        if vm.security_profile is not None:
            new_result['security_profile'] = dict()
            if vm.security_profile.encryption_at_host is not None:
                new_result['security_profile']['encryption_at_host'] = vm.security_profile.encryption_at_host
            if vm.security_profile.security_type is not None:
                new_result['security_profile']['security_type'] = vm.security_profile.security_type
            if vm.security_profile.uefi_settings is not None:
                new_result['security_profile']['uefi_settings'] = dict()
                if vm.security_profile.uefi_settings.secure_boot_enabled is not None:
                    new_result['security_profile']['uefi_settings']['secure_boot_enabled'] = vm.security_profile.uefi_settings.secure_boot_enabled
                if vm.security_profile.uefi_settings.v_tpm_enabled is not None:
                    new_result['security_profile']['uefi_settings']['v_tpm_enabled'] = vm.security_profile.uefi_settings.v_tpm_enabled
        new_result['power_state'] = power_state
        new_result['display_status'] = display_status
        new_result['provisioning_state'] = vm.provisioning_state
        new_result['id'] = vm.id
        new_result['resource_group'] = resource_group
        new_result['name'] = vm.name
        new_result['state'] = 'present'
        new_result['location'] = vm.location
        new_result['vm_size'] = result['hardware_profile']['vm_size']
        new_result['proximityPlacementGroup'] = result.get('proximity_placement_group')
        new_result['zones'] = result.get('zones', None)
        os_profile = result.get('os_profile')
        if os_profile is not None:
            new_result['admin_username'] = os_profile.get('admin_username')
        image = result['storage_profile'].get('image_reference')
        if image is not None:
            if image.get('publisher', None) is not None:
                new_result['image'] = {'publisher': image['publisher'], 'sku': image['sku'], 'offer': image['offer'], 'version': image['version']}
            else:
                new_result['image'] = {'id': image.get('id', None)}
        new_result['boot_diagnostics'] = {'enabled': 'diagnostics_profile' in result and 'boot_diagnostics' in result['diagnostics_profile'] and result['diagnostics_profile']['boot_diagnostics']['enabled'] or False, 'storage_uri': 'diagnostics_profile' in result and 'boot_diagnostics' in result['diagnostics_profile'] and result['diagnostics_profile']['boot_diagnostics'].get('storageUri', None)}
        if new_result['boot_diagnostics']['enabled']:
            new_result['boot_diagnostics']['console_screenshot_uri'] = result['instance_view']['boot_diagnostics'].get('console_screenshot_blob_uri')
            new_result['boot_diagnostics']['serial_console_log_uri'] = result['instance_view']['boot_diagnostics'].get('serial_console_log_blob_uri')
        vhd = result['storage_profile']['os_disk'].get('vhd')
        if vhd is not None:
            url = urlparse(vhd['uri'])
            new_result['storage_account_name'] = url.netloc.split('.')[0]
            new_result['storage_container_name'] = url.path.split('/')[1]
            new_result['storage_blob_name'] = url.path.split('/')[-1]
        new_result['os_disk_caching'] = result['storage_profile']['os_disk']['caching']
        new_result['os_type'] = result['storage_profile']['os_disk']['os_type']
        new_result['data_disks'] = []
        disks = result['storage_profile']['data_disks']
        for disk_index in range(len(disks)):
            new_result['data_disks'].append({'lun': disks[disk_index].get('lun'), 'name': disks[disk_index].get('name'), 'disk_size_gb': disks[disk_index].get('disk_size_gb'), 'managed_disk_type': disks[disk_index].get('managed_disk', {}).get('storage_account_type'), 'managed_disk_id': disks[disk_index].get('managed_disk', {}).get('id'), 'caching': disks[disk_index].get('caching')})
        new_result['network_interface_names'] = []
        nics = result['network_profile']['network_interfaces']
        for nic_index in range(len(nics)):
            new_result['network_interface_names'].append(re.sub('.*networkInterfaces/', '', nics[nic_index]['id']))
        new_result['tags'] = vm.tags
        return new_result