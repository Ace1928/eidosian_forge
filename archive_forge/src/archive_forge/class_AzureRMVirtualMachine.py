from __future__ import absolute_import, division, print_function
import base64
import random
import re
import time
from ansible.module_utils.basic import to_native, to_bytes
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import (AzureRMModuleBase,
class AzureRMVirtualMachine(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), custom_data=dict(type='str'), state=dict(choices=['present', 'absent'], default='present', type='str'), location=dict(type='str'), short_hostname=dict(type='str'), vm_size=dict(type='str'), priority=dict(type='str', choices=['None', 'Spot']), eviction_policy=dict(type='str', choices=['Deallocate', 'Delete']), max_price=dict(type='float', default=-1), admin_username=dict(type='str'), admin_password=dict(type='str', no_log=True), ssh_password_enabled=dict(type='bool', default=True, no_log=False), ssh_public_keys=dict(type='list', elements='dict'), image=dict(type='raw'), availability_set=dict(type='str'), storage_account_name=dict(type='str', aliases=['storage_account']), storage_container_name=dict(type='str', aliases=['storage_container'], default='vhds'), storage_blob_name=dict(type='str', aliases=['storage_blob']), os_disk_caching=dict(type='str', aliases=['disk_caching'], choices=['ReadOnly', 'ReadWrite']), os_disk_size_gb=dict(type='int'), managed_disk_type=dict(type='str', choices=['Standard_LRS', 'StandardSSD_LRS', 'StandardSSD_ZRS', 'Premium_LRS', 'Premium_ZRS', 'UltraSSD_LRS']), os_disk_name=dict(type='str'), proximity_placement_group=dict(type='dict', options=proximity_placement_group_spec), os_type=dict(type='str', choices=['Linux', 'Windows'], default='Linux'), public_ip_allocation_method=dict(type='str', choices=['Dynamic', 'Static', 'Disabled'], default='Static', aliases=['public_ip_allocation']), open_ports=dict(type='list', elements='str'), network_interface_names=dict(type='list', aliases=['network_interfaces'], elements='raw'), remove_on_absent=dict(type='list', default=['all'], elements='str'), virtual_network_resource_group=dict(type='str'), virtual_network_name=dict(type='str', aliases=['virtual_network']), subnet_name=dict(type='str', aliases=['subnet']), created_nsg=dict(type='bool', default=True), allocated=dict(type='bool', default=True), restarted=dict(type='bool', default=False), started=dict(type='bool'), force=dict(type='bool', default=False), generalized=dict(type='bool', default=False), data_disks=dict(type='list', elements='dict', options=dict(lun=dict(type='int', required=True), disk_size_gb=dict(type='int'), managed_disk_type=dict(type='str', choices=['Standard_LRS', 'StandardSSD_LRS', 'StandardSSD_ZRS', 'Premium_LRS', 'Premium_ZRS', 'UltraSSD_LRS']), storage_account_name=dict(type='str'), storage_container_name=dict(type='str', default='vhds'), storage_blob_name=dict(type='str'), caching=dict(type='str', choices=['ReadOnly', 'ReadOnly']))), plan=dict(type='dict'), zones=dict(type='list', elements='str'), accept_terms=dict(type='bool', default=False), license_type=dict(type='str', choices=['Windows_Server', 'Windows_Client', 'RHEL_BYOS', 'SLES_BYOS']), vm_identity=dict(type='dict', options=managed_identity_spec), winrm=dict(type='list', elements='dict', options=dict(protocol=dict(type='str', required=True, choices=['http', 'https']), source_vault=dict(type='str'), certificate_url=dict(type='str'), certificate_store=dict(type='str'))), boot_diagnostics=dict(type='dict', options=dict(enabled=dict(type='bool'), type=dict(type='str', choices=['managed']), storage_account=dict(type='str'), resource_group=dict(type='str')), mutually_exclusive=[('type', 'storage_account'), ('type', 'resource_group')]), ephemeral_os_disk=dict(type='bool'), windows_config=dict(type='dict', options=windows_configuration_spec), linux_config=dict(type='dict', options=linux_configuration_spec), security_profile=dict(type='dict'))
        self.resource_group = None
        self.name = None
        self.custom_data = None
        self.state = None
        self.location = None
        self.short_hostname = None
        self.vm_size = None
        self.priority = None
        self.eviction_policy = None
        self.admin_username = None
        self.admin_password = None
        self.ssh_password_enabled = None
        self.ssh_public_keys = None
        self.image = None
        self.availability_set = None
        self.storage_account_name = None
        self.storage_container_name = None
        self.storage_blob_name = None
        self.os_type = None
        self.os_disk_caching = None
        self.os_disk_size_gb = None
        self.managed_disk_type = None
        self.os_disk_name = None
        self.proximity_placement_group = None
        self.network_interface_names = None
        self.remove_on_absent = set()
        self.tags = None
        self.force = None
        self.public_ip_allocation_method = None
        self.open_ports = None
        self.virtual_network_resource_group = None
        self.virtual_network_name = None
        self.subnet_name = None
        self.created_nsg = None
        self.allocated = None
        self.restarted = None
        self.started = None
        self.generalized = None
        self.differences = None
        self.data_disks = None
        self.plan = None
        self.accept_terms = None
        self.zones = None
        self.license_type = None
        self.vm_identity = None
        self.boot_diagnostics = None
        self.ephemeral_os_disk = None
        self.linux_config = None
        self.windows_config = None
        self.security_profile = None
        self.results = dict(changed=False, actions=[], powerstate_change=None, ansible_facts=dict(azure_vm=None))
        super(AzureRMVirtualMachine, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True)

    @property
    def boot_diagnostics_present(self):
        return self.boot_diagnostics is not None and self.boot_diagnostics.get('enabled') is not None

    def get_boot_diagnostics_storage_account(self, limited=False, vm_dict=None):
        """
        Get the boot diagnostics storage account.

        Arguments:
          - limited - if true, limit the logic to the boot_diagnostics storage account
                      this is used if initial creation of the VM has a stanza with
                      boot_diagnostics disabled, so we only create a storage account
                      if the user specifies a storage account name inside the boot_diagnostics
                      schema
          - vm_dict - if invoked on an update, this is the current state of the vm including
                      tags, like the default storage group tag '_own_sa_'.

        Normal behavior:
          - try the self.boot_diagnostics.storage_account field
          - if not there, try the self.storage_account_name field
          - if not there, use the default storage account

        If limited is True:
          - try the self.boot_diagnostics.storage_account field
          - if not there, None
        """
        bsa = None
        if self.boot_diagnostics is not None and self.boot_diagnostics.get('storage_account') is not None:
            if self.boot_diagnostics.get('resource_group') is not None:
                bsa = self.get_storage_account(self.boot_diagnostics['resource_group'], self.boot_diagnostics['storage_account'])
            else:
                bsa = self.get_storage_account(self.resource_group, self.boot_diagnostics['storage_account'])
        elif limited:
            return None
        elif self.storage_account_name:
            bsa = self.get_storage_account(self.resource_group, self.storage_account_name)
        else:
            bsa = self.create_default_storage_account(vm_dict=vm_dict)
        self.log('boot diagnostics storage account:')
        self.log(self.serialize_obj(bsa, 'StorageAccount'), pretty_print=True)
        return bsa

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.remove_on_absent = set([resource.lower() for resource in self.remove_on_absent])
        self.zones = [int(i) for i in self.zones] if self.zones else None
        changed = False
        powerstate_change = None
        results = dict()
        vm = None
        network_interfaces = []
        requested_storage_uri = None
        requested_vhd_uri = None
        data_disk_requested_vhd_uri = None
        disable_ssh_password = None
        vm_dict = None
        image_reference = None
        custom_image = False
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        self.location = normalize_location_name(self.location)
        if self.state == 'present':
            if self.vm_size and (not self.vm_size_is_valid()):
                self.fail('Parameter error: vm_size {0} is not valid for your subscription and location.'.format(self.vm_size))
            if self.network_interface_names:
                for nic_name in self.network_interface_names:
                    nic = self.parse_network_interface(nic_name)
                    network_interfaces.append(nic)
            if self.ssh_public_keys:
                msg = 'Parameter error: expecting ssh_public_keys to be a list of type dict where each dict contains keys: path, key_data.'
                for key in self.ssh_public_keys:
                    if not isinstance(key, dict):
                        self.fail(msg)
                    if not key.get('path') or not key.get('key_data'):
                        self.fail(msg)
            if self.image and isinstance(self.image, dict):
                if all((key in self.image for key in ('publisher', 'offer', 'sku', 'version'))):
                    marketplace_image = self.get_marketplace_image_version()
                    if self.image['version'] == 'latest':
                        self.image['version'] = marketplace_image.name
                        self.log('Using image version {0}'.format(self.image['version']))
                    image_reference = self.compute_models.ImageReference(publisher=self.image['publisher'], offer=self.image['offer'], sku=self.image['sku'], version=self.image['version'])
                elif self.image.get('name'):
                    custom_image = True
                    image_reference = self.get_custom_image_reference(self.image.get('name'), self.image.get('resource_group'))
                elif self.image.get('id'):
                    try:
                        image_reference = self.compute_models.ImageReference(id=self.image['id'])
                    except Exception as exc:
                        self.fail('id Error: Cannot get image from the reference id - {0}'.format(self.image['id']))
                else:
                    self.fail('parameter error: expecting image to contain [publisher, offer, sku, version], [name, resource_group] or [id]')
            elif self.image and isinstance(self.image, str):
                custom_image = True
                image_reference = self.get_custom_image_reference(self.image)
            elif self.image:
                self.fail('parameter error: expecting image to be a string or dict not {0}'.format(type(self.image).__name__))
            if self.plan:
                if not self.plan.get('name') or not self.plan.get('product') or (not self.plan.get('publisher')):
                    self.fail('parameter error: plan must include name, product, and publisher')
            if not self.storage_blob_name and (not self.managed_disk_type):
                self.storage_blob_name = self.name + '.vhd'
            elif self.managed_disk_type:
                self.storage_blob_name = self.name
            if self.storage_account_name and (not self.managed_disk_type):
                properties = self.get_storage_account(self.resource_group, self.storage_account_name)
                requested_storage_uri = properties.primary_endpoints.blob
                requested_vhd_uri = '{0}{1}/{2}'.format(requested_storage_uri, self.storage_container_name, self.storage_blob_name)
            disable_ssh_password = not self.ssh_password_enabled
        try:
            self.log('Fetching virtual machine {0}'.format(self.name))
            vm = self.compute_client.virtual_machines.get(self.resource_group, self.name, expand='instanceview')
            retry_count = 0
            while True:
                if retry_count == 10:
                    self.fail('Error {0} has a provisioning state of Updating. Expecting state to be Successed.'.format(self.name))
                if vm.provisioning_state == 'Updating':
                    retry_count = retry_count + 1
                    time.sleep(300)
                    vm = self.compute_client.virtual_machines.get(self.resource_group, self.name, expand='instanceview')
                else:
                    break
            vm_dict = self.serialize_vm(vm)
            if self.state == 'present':
                differences = []
                current_nics = []
                results = vm_dict
                current_osdisk = vm_dict['storage_profile']['os_disk']
                current_ephemeral = current_osdisk.get('diff_disk_ettings', None)
                current_properties = vm_dict
                if self.priority and self.priority != current_properties.get('priority', 'None'):
                    self.fail('VM Priority is not updatable: requested virtual machine priority is {0}'.format(self.priority))
                if self.eviction_policy and self.eviction_policy != current_properties.get('eviction_policy', None):
                    self.fail('VM Eviction Policy is not updatable: requested virtual machine eviction policy is {0}'.format(self.eviction_policy))
                if self.max_price and vm_dict.get('billing_profile', None) and (self.max_price != vm_dict['billing_profile'].get('max_price', None)):
                    self.fail('VM Maximum Price is not updatable: requested virtual machine maximum price is {0}'.format(self.max_price))
                if self.ephemeral_os_disk and current_ephemeral is None:
                    self.fail('Ephemeral OS disk not updatable: virtual machine ephemeral OS disk is {0}'.format(self.ephemeral_os_disk))
                elif not self.ephemeral_os_disk and current_ephemeral is not None:
                    self.fail('Ephemeral OS disk not updatable: virtual machine ephemeral OS disk is {0}'.format(self.ephemeral_os_disk))
                if self.network_interface_names:
                    for nic in vm_dict['network_profile']['network_interfaces']:
                        current_nics.append(nic['id'])
                    if set(current_nics) != set(network_interfaces):
                        self.log('CHANGED: virtual machine {0} - network interfaces are different.'.format(self.name))
                        differences.append('Network Interfaces')
                        updated_nics = [dict(id=id, primary=i == 0) for i, id in enumerate(network_interfaces)]
                        vm_dict['network_profile']['network_interfaces'] = updated_nics
                        changed = True
                if self.os_disk_caching and self.os_disk_caching != vm_dict['storage_profile']['os_disk']['caching']:
                    self.log('CHANGED: virtual machine {0} - OS disk caching'.format(self.name))
                    differences.append('OS Disk caching')
                    changed = True
                    vm_dict['storage_profile']['os_disk']['caching'] = self.os_disk_caching
                if self.os_disk_name and self.os_disk_name != vm_dict['storage_profile']['os_disk']['name']:
                    self.log('CHANGED: virtual machine {0} - OS disk name'.format(self.name))
                    differences.append('OS Disk name')
                    changed = True
                    vm_dict['storage_profile']['os_disk']['name'] = self.os_disk_name
                if self.os_disk_size_gb and self.os_disk_size_gb != vm_dict['storage_profile']['os_disk'].get('disk_size_gb'):
                    self.log('CHANGED: virtual machine {0} - OS disk size '.format(self.name))
                    differences.append('OS Disk size')
                    changed = True
                    vm_dict['storage_profile']['os_disk']['disk_size_gb'] = self.os_disk_size_gb
                if self.vm_size and self.vm_size != vm_dict['hardware_profile']['vm_size']:
                    self.log('CHANGED: virtual machine {0} - size '.format(self.name))
                    differences.append('VM size')
                    changed = True
                    vm_dict['hardware_profile']['vm_size'] = self.vm_size
                update_tags, vm_dict['tags'] = self.update_tags(vm_dict.get('tags', dict()))
                if update_tags:
                    differences.append('Tags')
                    changed = True
                if self.short_hostname and self.short_hostname != vm_dict['os_profile']['computer_name']:
                    self.log('CHANGED: virtual machine {0} - short hostname'.format(self.name))
                    differences.append('Short Hostname')
                    changed = True
                    vm_dict['os_orofile']['computer_name'] = self.short_hostname
                if self.started and vm_dict['powerstate'] not in ['starting', 'running'] and self.allocated:
                    self.log("CHANGED: virtual machine {0} not running and requested state 'running'".format(self.name))
                    changed = True
                    powerstate_change = 'poweron'
                elif self.state == 'present' and vm_dict['powerstate'] == 'running' and self.restarted:
                    self.log("CHANGED: virtual machine {0} {1} and requested state 'restarted'".format(self.name, vm_dict['powerstate']))
                    changed = True
                    powerstate_change = 'restarted'
                elif self.state == 'present' and (not self.allocated) and (vm_dict['powerstate'] not in ['deallocated', 'deallocating']):
                    self.log("CHANGED: virtual machine {0} {1} and requested state 'deallocated'".format(self.name, vm_dict['powerstate']))
                    changed = True
                    powerstate_change = 'deallocated'
                elif self.started is not None and (not self.started) and (vm_dict['powerstate'] == 'running'):
                    self.log("CHANGED: virtual machine {0} running and requested state 'stopped'".format(self.name))
                    changed = True
                    powerstate_change = 'poweroff'
                elif self.generalized and vm_dict['powerstate'] != 'generalized':
                    self.log("CHANGED: virtual machine {0} requested to be 'generalized'".format(self.name))
                    changed = True
                    powerstate_change = 'generalized'
                vm_dict['zones'] = [int(i) for i in vm_dict['zones']] if 'zones' in vm_dict and vm_dict['zones'] else None
                if self.zones is not None and self.zones != vm_dict['zones']:
                    self.log('CHANGED: virtual machine {0} zones'.format(self.name))
                    differences.append('Zones')
                    changed = True
                if self.license_type is not None and vm_dict.get('licenseType') != self.license_type:
                    differences.append('License Type')
                    changed = True
                if self.vm_identity:
                    update_vm_identity = False
                    if 'None' in self.vm_identity.get('type') and 'identity' not in vm_dict:
                        pass
                    elif 'identity' not in vm_dict:
                        update_vm_identity = True
                    elif vm_dict['identity']['type'] != self.vm_identity.get('type'):
                        update_vm_identity = True
                    elif 'UserAssigned' in self.vm_identity.get('type'):
                        new_managed_identities = set(self.vm_identity.get('user_assigned_identities', {}).get('id', []))
                        current_managed_identities = set(vm_dict['identity']['user_assigned_identities'].keys())
                        if self.vm_identity.get('user_assigned_identities', {}).get('append', False) is True:
                            if len(current_managed_identities) != len(new_managed_identities.union(current_managed_identities)):
                                update_vm_identity = True
                        elif current_managed_identities.difference(new_managed_identities) != set():
                            update_vm_identity = True
                    if update_vm_identity:
                        differences.append('Managed Identities')
                        changed = True
                if self.security_profile is not None:
                    update_security_profile = False
                    if 'securityProfile' not in vm_dict.keys():
                        update_security_profile = True
                        differences.append('security_profile')
                    else:
                        if self.security_profile.get('encryption_at_host') is not None:
                            if bool(self.security_profile.get('encryption_at_host')) != bool(vm_dict['security_profile']['encryption_at_host']):
                                update_security_profile = True
                            else:
                                self.security_profile['encryption_at_host'] = vm_dict['security_profile']['encryption_at_host']
                        if self.security_profile.get('security_type') is not None:
                            if self.security_profile.get('security_type') != vm_dict['security_profile']['security_type']:
                                update_security_profile = True
                        if self.security_profile.get('uefi_settings') is not None:
                            if self.security_profile['uefi_settings'].get('secure_boot_enabled') is not None:
                                if bool(self.security_profile['uefi_settings']['secure_boot_enabled']) != bool(vm_dict['security_profile']['uefi_settings']['secure_boot_enabled']):
                                    update_security_profile = True
                            else:
                                self.security_profile['uefi_settings']['secure_boot_enabled'] = vm_dict['security_profile']['uefi_settings']['secure_soot_enabled']
                            if self.security_profile['uefi_settings'].get('v_tpm_enabled') is not None:
                                if bool(self.security_profile['uefi_settings']['v_tpm_enabled']) != bool(vm_dict['security_profile']['uefi_settings']['v_tpm_enabled']):
                                    update_security_profile = True
                            else:
                                self.security_profile['uefi_settings']['v_tpm_enabled'] = vm_dict['security_profile']['uefi_settings']['v_tpm_enabled']
                    if update_security_profile:
                        changed = True
                        differences.append('security_profile')
                if self.windows_config is not None and vm_dict['os_profile'].get('windows_configuration') is not None:
                    if self.windows_config['enable_automatic_updates'] != vm_dict['os_profile']['windows_configuration']['enable_automatic_updates']:
                        self.fail("(PropertyChangeNotAllowed) Changing property 'windowsConfiguration.enableAutomaticUpdates' is not allowed.")
                    if self.windows_config['provision_vm_agent'] != vm_dict['os_profile']['windows_configuration']['provision_vm_agent']:
                        self.fail("(PropertyChangeNotAllowed) Changing property 'windowsConfiguration.provisionVMAgent' is not allowed.")
                if self.linux_config is not None and vm_dict['os_profile'].get('linux_configuration') is not None:
                    if self.linux_config['disable_password_authentication'] != vm_dict['os_profile']['linux_configuration']['disable_password_authentication']:
                        self.fail("(PropertyChangeNotAllowed) Changing property 'linuxConfiguration.disablePasswordAuthentication' is not allowed.")
                if 'diagnostics_profile' not in vm_dict:
                    vm_dict['diagnostics_profile'] = {}
                if 'boot_diagnostics' not in vm_dict['diagnostics_profile']:
                    vm_dict['diagnostics_profile']['boot_diagnostics'] = {'enabled': False, 'storageUri': None}
                if self.boot_diagnostics_present:
                    current_boot_diagnostics = vm_dict['diagnostics_profile']['boot_diagnostics']
                    boot_diagnostics_changed = False
                    if self.boot_diagnostics['enabled'] != current_boot_diagnostics['enabled']:
                        current_boot_diagnostics['enabled'] = self.boot_diagnostics['enabled']
                        boot_diagnostics_changed = True
                    if self.boot_diagnostics.get('type') is not None and self.boot_diagnostics['type'] == 'managed':
                        boot_diagnostics_blob = None
                    else:
                        boot_diagnostics_storage_account = self.get_boot_diagnostics_storage_account(limited=not self.boot_diagnostics['enabled'], vm_dict=vm_dict)
                        boot_diagnostics_blob = boot_diagnostics_storage_account.primary_endpoints.blob if boot_diagnostics_storage_account else None
                    if current_boot_diagnostics.get('storageUri') != boot_diagnostics_blob:
                        current_boot_diagnostics['storageUri'] = boot_diagnostics_blob
                        boot_diagnostics_changed = True
                    if boot_diagnostics_changed:
                        differences.append('Boot Diagnostics')
                        changed = True
                    own_sa = (self.tags or {}).get('_own_sa_', None)
                    cur_sa = vm_dict.get('tags', {}).get('_own_sa_', None)
                    if own_sa and own_sa != cur_sa:
                        if 'Tags' not in differences:
                            differences.append('Tags')
                        if 'tags' not in vm_dict:
                            vm_dict['tags'] = {}
                        vm_dict['tags']['_own_sa_'] = own_sa
                        changed = True
                self.differences = differences
            elif self.state == 'absent':
                self.log("CHANGED: virtual machine {0} exists and requested state is 'absent'".format(self.name))
                results = dict()
                changed = True
        except ResourceNotFoundError:
            self.log('Virtual machine {0} does not exist'.format(self.name))
            if self.state == 'present':
                self.log("CHANGED: virtual machine {0} does not exist but state is 'present'.".format(self.name))
                changed = True
        self.results['changed'] = changed
        self.results['ansible_facts']['azure_vm'] = results
        self.results['powerstate_change'] = powerstate_change
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                if not vm:
                    self.log('Create virtual machine {0}'.format(self.name))
                    self.results['actions'].append('Created VM {0}'.format(self.name))
                    if not self.admin_username:
                        self.fail('Parameter error: admin_username required when creating a virtual machine.')
                    if self.os_type == 'Linux':
                        if disable_ssh_password and (not self.ssh_public_keys):
                            self.fail('Parameter error: ssh_public_keys required when disabling SSH password.')
                    if not image_reference:
                        self.fail('Parameter error: an image is required when creating a virtual machine.')
                    availability_set_resource = None
                    if self.availability_set:
                        parsed_availability_set = parse_resource_id(self.availability_set)
                        availability_set = self.get_availability_set(parsed_availability_set.get('resource_group', self.resource_group), parsed_availability_set.get('name'))
                        availability_set_resource = self.compute_models.SubResource(id=availability_set.id)
                        if self.zones:
                            self.fail("Parameter error: you can't use Availability Set and Availability Zones at the same time")
                    proximity_placement_group_resource = None
                    if self.proximity_placement_group is not None:
                        if self.proximity_placement_group.get('id') is not None:
                            proximity_placement_group_resource = self.compute_models.SubResource(id=self.proximity_placement_group['id'])
                        elif self.proximity_placement_group.get('name') is not None and self.proximity_placement_group.get('resource_group') is not None:
                            proximity_placement_group = self.get_proximity_placement_group(self.proximity_placement_group.get('resource_group'), self.proximity_placement_group.get('name'))
                            proximity_placement_group_resource = self.compute_models.SubResource(id=proximity_placement_group.id)
                        else:
                            self.fail('Parameter error: Please recheck your proximity placement group ')
                    if not self.network_interface_names:
                        default_nic = self.create_default_nic()
                        self.log('network interface:')
                        self.log(self.serialize_obj(default_nic, 'NetworkInterface'), pretty_print=True)
                        network_interfaces = [default_nic.id]
                    if not self.storage_account_name and (not self.managed_disk_type):
                        storage_account = self.create_default_storage_account()
                        self.log('os disk storage account:')
                        self.log(self.serialize_obj(storage_account, 'StorageAccount'), pretty_print=True)
                        requested_storage_uri = 'https://{0}.blob.{1}/'.format(storage_account.name, self._cloud_environment.suffixes.storage_endpoint)
                        requested_vhd_uri = '{0}{1}/{2}'.format(requested_storage_uri, self.storage_container_name, self.storage_blob_name)
                    if not self.os_disk_caching:
                        self.os_disk_caching = 'ReadOnly'
                    if not self.short_hostname:
                        self.short_hostname = self.name
                    nics = [self.compute_models.NetworkInterfaceReference(id=id, primary=i == 0) for i, id in enumerate(network_interfaces)]
                    if self.managed_disk_type:
                        vhd = None
                        managed_disk = self.compute_models.ManagedDiskParameters(storage_account_type=self.managed_disk_type)
                    elif custom_image:
                        vhd = None
                        managed_disk = None
                    else:
                        vhd = self.compute_models.VirtualHardDisk(uri=requested_vhd_uri)
                        managed_disk = None
                    plan = None
                    if self.plan:
                        plan = self.compute_models.Plan(name=self.plan.get('name'), product=self.plan.get('product'), publisher=self.plan.get('publisher'), promotion_code=self.plan.get('promotion_code'))
                    if self.boot_diagnostics_present and self.boot_diagnostics['enabled'] and (self.boot_diagnostics.get('type') != 'managed'):
                        boot_diag_storage_account = self.get_boot_diagnostics_storage_account()
                    vm_resource = self.compute_models.VirtualMachine(location=self.location, tags=self.tags, os_profile=self.compute_models.OSProfile(admin_username=self.admin_username, computer_name=self.short_hostname), hardware_profile=self.compute_models.HardwareProfile(vm_size=self.vm_size), storage_profile=self.compute_models.StorageProfile(os_disk=self.compute_models.OSDisk(name=self.os_disk_name if self.os_disk_name else self.storage_blob_name, vhd=vhd, managed_disk=managed_disk, create_option=self.compute_models.DiskCreateOptionTypes.from_image, caching=self.os_disk_caching, disk_size_gb=self.os_disk_size_gb, diff_disk_settings=self.compute_models.DiffDiskSettings(option='Local') if self.ephemeral_os_disk else None), image_reference=image_reference), network_profile=self.compute_models.NetworkProfile(network_interfaces=nics), availability_set=availability_set_resource, proximity_placement_group=proximity_placement_group_resource, plan=plan, zones=self.zones)
                    if self.priority == 'Spot':
                        vm_resource.priority = self.priority
                        vm_resource.eviction_policy = self.eviction_policy
                        vm_resource.billing_profile = self.compute_models.BillingProfile(max_price=self.max_price)
                    if self.license_type is not None:
                        vm_resource.license_type = self.license_type
                    if self.vm_identity:
                        if 'UserAssigned' in self.vm_identity.get('type') and len(self.vm_identity.get('user_assigned_identities', {}).get('id', [])) != 0:
                            user_assigned_identities_dict = {uami: dict() for uami in self.vm_identity.get('user_assigned_identities').get('id')}
                            vm_resource.identity = self.compute_models.VirtualMachineIdentity(type=self.vm_identity.get('type'), user_assigned_identities=user_assigned_identities_dict)
                        elif 'UserAssigned' in self.vm_identity.get('type') and len(self.vm_identity.get('user_assigned_identities', {}).get('id', [])) == 0:
                            self.fail('UserAssigned specified but no User Identity IDs provided')
                        else:
                            vm_resource.identity = self.compute_models.VirtualMachineIdentity(type=self.vm_identity.get('type'))
                    if self.winrm:
                        winrm_listeners = list()
                        for winrm_listener in self.winrm:
                            winrm_listeners.append(self.compute_models.WinRMListener(protocol=winrm_listener.get('protocol'), certificate_url=winrm_listener.get('certificate_url')))
                            if winrm_listener.get('source_vault'):
                                if not vm_resource.os_profile.secrets:
                                    vm_resource.os_profile.secrets = list()
                                vm_resource.os_profile.secrets.append(self.compute_models.VaultSecretGroup(source_vault=self.compute_models.SubResource(id=winrm_listener.get('source_vault')), vault_certificates=[self.compute_models.VaultCertificate(certificate_url=winrm_listener.get('certificate_url'), certificate_store=winrm_listener.get('certificate_store'))]))
                        self.winrm = self.compute_models.WinRMConfiguration(listeners=winrm_listeners)
                    if self.os_type == 'Windows':
                        vm_resource.os_profile.windows_configuration = self.compute_models.WindowsConfiguration(win_rm=self.winrm, provision_vm_agent=self.windows_config['provision_vm_agent'] if self.windows_config is not None else True, enable_automatic_updates=self.windows_config['enable_automatic_updates'] if self.windows_config is not None else True)
                    if self.boot_diagnostics_present:
                        if self.boot_diagnostics['enabled'] and self.boot_diagnostics.get('type') != 'managed':
                            storage_uri = boot_diag_storage_account.primary_endpoints.blob
                        else:
                            storage_uri = None
                        vm_resource.diagnostics_profile = self.compute_models.DiagnosticsProfile(boot_diagnostics=self.compute_models.BootDiagnostics(enabled=self.boot_diagnostics['enabled'], storage_uri=storage_uri))
                    if self.admin_password:
                        vm_resource.os_profile.admin_password = self.admin_password
                    if self.custom_data:
                        vm_resource.os_profile.custom_data = to_native(base64.b64encode(to_bytes(self.custom_data)))
                    if self.os_type == 'Linux':
                        vm_resource.os_profile.linux_configuration = self.compute_models.LinuxConfiguration(disable_password_authentication=self.linux_config['disable_password_authentication'] if self.linux_config else disable_ssh_password)
                    if self.ssh_public_keys:
                        ssh_config = self.compute_models.SshConfiguration()
                        ssh_config.public_keys = [self.compute_models.SshPublicKey(path=key['path'], key_data=key['key_data']) for key in self.ssh_public_keys]
                        vm_resource.os_profile.linux_configuration.ssh = ssh_config
                    if self.data_disks:
                        data_disks = []
                        count = 0
                        for data_disk in self.data_disks:
                            if not data_disk.get('managed_disk_type'):
                                if not data_disk.get('storage_blob_name'):
                                    data_disk['storage_blob_name'] = self.name + '-data-' + str(count) + '.vhd'
                                    count += 1
                                if data_disk.get('storage_account_name'):
                                    data_disk_storage_account = self.get_storage_account(self.resource_group, data_disk['storage_account_name'])
                                else:
                                    data_disk_storage_account = self.create_default_storage_account()
                                    self.log('data disk storage account:')
                                    self.log(self.serialize_obj(data_disk_storage_account, 'StorageAccount'), pretty_print=True)
                                if not data_disk.get('storage_container_name'):
                                    data_disk['storage_container_name'] = 'vhds'
                                data_disk_requested_vhd_uri = 'https://{0}.blob.{1}/{2}/{3}'.format(data_disk_storage_account.name, self._cloud_environment.suffixes.storage_endpoint, data_disk['storage_container_name'], data_disk['storage_blob_name'])
                            if not data_disk.get('managed_disk_type'):
                                data_disk_managed_disk = None
                                disk_name = data_disk['storage_blob_name']
                                data_disk_vhd = self.compute_models.VirtualHardDisk(uri=data_disk_requested_vhd_uri)
                            else:
                                data_disk_vhd = None
                                data_disk_managed_disk = self.compute_models.ManagedDiskParameters(storage_account_type=data_disk['managed_disk_type'])
                                disk_name = self.name + '-datadisk-' + str(count)
                                count += 1
                            data_disk['caching'] = data_disk.get('caching', 'ReadOnly')
                            data_disks.append(self.compute_models.DataDisk(lun=data_disk['lun'], name=disk_name, vhd=data_disk_vhd, caching=data_disk['caching'], create_option=self.compute_models.DiskCreateOptionTypes.empty, disk_size_gb=data_disk['disk_size_gb'], managed_disk=data_disk_managed_disk))
                        vm_resource.storage_profile.data_disks = data_disks
                    if self.accept_terms is True:
                        if not self.plan or not all([self.plan.get('name'), self.plan.get('product'), self.plan.get('publisher')]):
                            self.fail('parameter error: plan must be specified and include name, product, and publisher')
                        try:
                            plan_name = self.plan.get('name')
                            plan_product = self.plan.get('product')
                            plan_publisher = self.plan.get('publisher')
                            term = self.marketplace_client.marketplace_agreements.get(offer_type='virtualmachine', publisher_id=plan_publisher, offer_id=plan_product, plan_id=plan_name)
                            term.accepted = True
                            self.marketplace_client.marketplace_agreements.create(offer_type='virtualmachine', publisher_id=plan_publisher, offer_id=plan_product, plan_id=plan_name, parameters=term)
                        except Exception as exc:
                            self.fail(('Error accepting terms for virtual machine {0} with plan {1}. ' + 'Only service admin/account admin users can purchase images ' + 'from the marketplace. - {2}').format(self.name, self.plan, str(exc)))
                    if self.security_profile is not None:
                        uefi_settings_spec = None
                        if self.security_profile.get('uefi_settings') is not None:
                            uefi_settings_spec = self.compute_models.UefiSettings(secure_boot_enabled=self.security_profile['uefi_settings'].get('secure_boot_enabled'), v_tpm_enabled=self.security_profile['uefi_settings'].get('v_tpm_enabled'))
                        security_profile = self.compute_models.SecurityProfile(uefi_settings=uefi_settings_spec, encryption_at_host=self.security_profile.get('encryption_at_host'), security_type=self.security_profile.get('security_type'))
                        vm_resource.security_profile = security_profile
                    self.log('Create virtual machine with parameters:')
                    self.create_or_update_vm(vm_resource, 'all_autocreated' in self.remove_on_absent)
                elif self.differences and len(self.differences) > 0:
                    self.log('Update virtual machine {0}'.format(self.name))
                    self.results['actions'].append('Updated VM {0}'.format(self.name))
                    nics = [self.compute_models.NetworkInterfaceReference(id=interface['id'], primary=i == 0) for i, interface in enumerate(vm_dict['network_profile']['network_interfaces'])]
                    if not vm_dict['storage_profile']['os_disk'].get('managedDisk'):
                        managed_disk = None
                        vhd = self.compute_models.VirtualHardDisk(uri=vm_dict['storage_profile']['os_disk'].get('vhd', {}).get('uri'))
                    else:
                        vhd = None
                        managed_disk = self.compute_models.ManagedDiskParameters(storage_account_type=vm_dict['storage_profile']['os_disk']['managed_disk'].get('storage_account_type'))
                    proximity_placement_group_resource = None
                    try:
                        proximity_placement_group_resource = self.compute_models.SubResource(id=vm_dict['proximity_placement_group'].get('id'))
                    except Exception:
                        pass
                    availability_set_resource = None
                    try:
                        availability_set_resource = self.compute_models.SubResource(id=vm_dict['availability_set'].get('id'))
                    except Exception:
                        pass
                    if 'imageReference' in vm_dict['storage_profile'].keys():
                        if 'id' in vm_dict['storage_profile']['image_reference'].keys():
                            image_reference = self.compute_models.ImageReference(id=vm_dict['storage_profile']['image_reference']['id'])
                        else:
                            image_reference = self.compute_models.ImageReference(publisher=vm_dict['storage_profile']['image_reference'].get('publisher'), offer=vm_dict['storage_profile']['image_reference'].get('offer'), sku=vm_dict['storage_profile']['image_reference'].get('sku'), version=vm_dict['storage_profile']['image_reference'].get('version'))
                    else:
                        image_reference = None
                    if self.zones is not None and vm_dict['zones'] != self.zones:
                        self.fail("You can't change the Availability Zone of a virtual machine (have: {0}, want: {1})".format(vm_dict['zones'], self.zones))
                    if 'os_profile' in vm_dict:
                        os_profile = self.compute_models.OSProfile(admin_username=vm_dict.get('os_profile', {}).get('admin_username'), computer_name=vm_dict.get('os_profile', {}).get('computer_name'))
                    else:
                        os_profile = None
                    vm_resource = self.compute_models.VirtualMachine(location=vm_dict['location'], os_profile=os_profile, hardware_profile=self.compute_models.HardwareProfile(vm_size=vm_dict['hardware_profile'].get('vm_size')), storage_profile=self.compute_models.StorageProfile(os_disk=self.compute_models.OSDisk(name=vm_dict['storage_profile']['os_disk'].get('name'), vhd=vhd, managed_disk=managed_disk, create_option=vm_dict['storage_profile']['os_disk'].get('create_option'), os_type=vm_dict['storage_profile']['os_disk'].get('os_type'), caching=vm_dict['storage_profile']['os_disk'].get('caching'), disk_size_gb=vm_dict['storage_profile']['os_disk'].get('disk_size_gb')), image_reference=image_reference), availability_set=availability_set_resource, proximity_placement_group=proximity_placement_group_resource, network_profile=self.compute_models.NetworkProfile(network_interfaces=nics))
                    if self.license_type is not None:
                        vm_resource.license_type = self.license_type
                    if self.vm_identity is not None:
                        if self.vm_identity.get('user_assigned_identities', {}) is not None and self.vm_identity.get('user_assigned_identities', {}).get('append', False) is True:
                            if 'identity' in vm_dict and 'user_assigned_identities' in vm_dict['identity']:
                                current_user_assigned_identities_dict = {uami: dict() for uami in vm_dict['identity']['user_assigned_identities'].keys()}
                                vm_identity_user_assigned_append = True
                            else:
                                vm_identity_user_assigned_append = False
                        else:
                            vm_identity_user_assigned_append = False
                        if 'UserAssigned' in self.vm_identity.get('type') and len(self.vm_identity.get('user_assigned_identities', {}).get('id', [])) != 0:
                            user_assigned_identities_dict = {uami: dict() for uami in self.vm_identity.get('user_assigned_identities').get('id')}
                            if vm_identity_user_assigned_append:
                                user_assigned_identities_dict = {**user_assigned_identities_dict, **current_user_assigned_identities_dict}
                            vm_resource.identity = self.compute_models.VirtualMachineIdentity(type=self.vm_identity.get('type'), user_assigned_identities=user_assigned_identities_dict)
                        elif 'UserAssigned' in self.vm_identity.get('type') and len(self.vm_identity.get('user_assigned_identities', {}).get('id', [])) == 0:
                            if vm_identity_user_assigned_append is False:
                                self.fail('UserAssigned specified but no User Assigned IDs provided' + ' and no UserAssigned identities are currently assigned to the VM')
                            else:
                                vm_resource.identity = self.compute_models.VirtualMachineIdentity(type=self.vm_identity.get('type'), user_assigned_identities=current_user_assigned_identities_dict)
                        else:
                            vm_resource.identity = self.compute_models.VirtualMachineIdentity(type=self.vm_identity.get('type'))
                    if self.boot_diagnostics is not None:
                        storage_uri = None
                        if 'storageUri' in vm_dict['diagnostics_profile']['boot_diagnostics']:
                            storage_uri = vm_dict['diagnostics_profile']['boot_diagnostics']['storageUri']
                        vm_resource.diagnostics_profile = self.compute_models.DiagnosticsProfile(boot_diagnostics=self.compute_models.BootDiagnostics(enabled=vm_dict['diagnostics_profile']['boot_diagnostics']['enabled'], storage_uri=storage_uri))
                    if vm_dict.get('tags'):
                        vm_resource.tags = vm_dict['tags']
                    if vm_dict.get('os_profile', {}).get('custom_data'):
                        custom_data = vm_dict['os_profile']['custom_data']
                        vm_resource.os_profile.custom_data = to_native(base64.b64encode(to_bytes(custom_data)))
                    if vm_dict.get('os_profile', {}).get('admin_password'):
                        vm_resource.os_profile.admin_password = vm_dict['os_profile']['admin_password']
                    windows_config = vm_dict.get('os_profile', {}).get('windows_configuration')
                    if windows_config:
                        if self.windows_config is not None:
                            vm_resource.os_profile.windows_configuration = self.compute_models.WindowsConfiguration(provision_vm_agent=self.windows_config['provision_vm_agent'], enable_automatic_updates=self.windows_config['enable_automatic_updates'])
                        else:
                            vm_resource.os_profile.windows_configuration = self.compute_models.WindowsConfiguration(provision_vm_agent=windows_config.get('provisionVMAgent', True), enable_automatic_updates=windows_config.get('enableAutomaticUpdates', True))
                    linux_config = vm_dict.get('os_profile', {}).get('linux_configuration')
                    if linux_config:
                        if self.linux_config is not None:
                            vm_resource.os_profile.linux_configuration = self.compute_models.LinuxConfiguration(disable_password_authentication=self.linux_config['disable_password_authentication'])
                        else:
                            vm_resource.os_profile.linux_configuration = self.compute_models.LinuxConfiguration(disable_password_authentication=linux_config.get('disable_password_authentication', False))
                        ssh_config = linux_config.get('ssh', None)
                        if ssh_config:
                            public_keys = ssh_config.get('publicKeys')
                            if public_keys:
                                vm_resource.os_profile.linux_configuration.ssh = self.compute_models.SshConfiguration(public_keys=[])
                                for key in public_keys:
                                    vm_resource.os_profile.linux_configuration.ssh.public_keys.append(self.compute_models.SshPublicKey(path=key['path'], key_data=key['keyData']))
                    if vm_dict['storage_profile'].get('data_disks'):
                        data_disks = []
                        for data_disk in vm_dict['storage_profile']['data_disks']:
                            if data_disk.get('managed_disk'):
                                managed_disk_type = data_disk['managed_disk'].get('storage_account_type')
                                data_disk_managed_disk = self.compute_models.ManagedDiskParameters(storage_account_type=managed_disk_type)
                                data_disk_vhd = None
                            else:
                                data_disk_vhd = data_disk['vhd']['uri']
                                data_disk_managed_disk = None
                            data_disks.append(self.compute_models.DataDisk(lun=int(data_disk['lun']), name=data_disk.get('name'), vhd=data_disk_vhd, caching=data_disk.get('caching'), create_option=data_disk.get('create_option'), disk_size_gb=int(data_disk.get('disk_size_gb', 0)) or None, managed_disk=data_disk_managed_disk))
                        vm_resource.storage_profile.data_disks = data_disks
                    if self.security_profile is not None:
                        uefi_settings_spec = None
                        if self.security_profile.get('uefi_settings') is not None:
                            uefi_settings_spec = self.compute_models.UefiSettings(secure_boot_enabled=self.security_profile['uefi_settings'].get('secure_boot_enabled'), v_tpm_enabled=self.security_profile['uefi_settings'].get('v_tpm_enabled'))
                        security_profile = self.compute_models.SecurityProfile(uefi_settings=uefi_settings_spec, encryption_at_host=self.security_profile.get('encryption_at_host'), security_type=self.security_profile.get('security_type'))
                        vm_resource.security_profile = security_profile
                    self.log('Update virtual machine with parameters:')
                    self.create_or_update_vm(vm_resource, False)
                if powerstate_change == 'poweron' and self.results['ansible_facts']['azure_vm']['powerstate'] != 'running':
                    self.power_on_vm()
                elif powerstate_change == 'poweroff' and self.results['ansible_facts']['azure_vm']['powerstate'] == 'running':
                    self.power_off_vm()
                elif powerstate_change == 'restarted':
                    self.restart_vm()
                elif powerstate_change == 'deallocated':
                    self.deallocate_vm()
                elif powerstate_change == 'generalized':
                    self.power_off_vm()
                    self.generalize_vm()
                self.results['ansible_facts']['azure_vm'] = self.serialize_vm(self.get_vm())
            elif self.state == 'absent':
                self.log('Delete virtual machine {0}'.format(self.name))
                self.results['ansible_facts']['azure_vm'] = None
                self.delete_vm(vm)
        del self.results['actions']
        return self.results

    def get_vm(self):
        """
        Get the VM with expanded instanceView

        :return: VirtualMachine object
        """
        try:
            vm = self.compute_client.virtual_machines.get(self.resource_group, self.name, expand='instanceview')
            return vm
        except Exception as exc:
            self.fail('Error getting virtual machine {0} - {1}'.format(self.name, str(exc)))

    def serialize_vm(self, vm):
        """
        Convert a VirtualMachine object to dict.

        :param vm: VirtualMachine object
        :return: dict
        """
        result = self.serialize_obj(vm, AZURE_OBJECT_CLASS, enum_modules=AZURE_ENUM_MODULES)
        result['id'] = vm.id
        result['name'] = vm.name
        result['type'] = vm.type
        result['location'] = vm.location
        result['tags'] = vm.tags
        result['powerstate'] = dict()
        if vm.instance_view:
            result['powerstate'] = next((s.code.replace('PowerState/', '') for s in vm.instance_view.statuses if s.code.startswith('PowerState')), None)
            for s in vm.instance_view.statuses:
                if s.code.lower() == 'osstate/generalized':
                    result['powerstate'] = 'generalized'
        for interface in vm.network_profile.network_interfaces:
            int_dict = azure_id_to_dict(interface.id)
            nic = self.get_network_interface(int_dict['resourceGroups'], int_dict['networkInterfaces'])
            for interface_dict in result['network_profile']['network_interfaces']:
                if interface_dict['id'] == interface.id:
                    nic_dict = self.serialize_obj(nic, 'NetworkInterface')
                    interface_dict['name'] = int_dict['networkInterfaces']
                    interface_dict['properties'] = nic_dict
        for interface in result['network_profile']['network_interfaces']:
            for config in interface['properties']['ip_configurations']:
                if config.get('public_ip_address'):
                    pipid_dict = azure_id_to_dict(config['public_ip_address']['id'])
                    try:
                        pip = self.network_client.public_ip_addresses.get(pipid_dict['resourceGroups'], pipid_dict['publicIPAddresses'])
                    except Exception as exc:
                        self.fail('Error fetching public ip {0} - {1}'.format(pipid_dict['publicIPAddresses'], str(exc)))
                    pip_dict = self.serialize_obj(pip, 'PublicIPAddress')
                    config['public_ip_address']['name'] = pipid_dict['publicIPAddresses']
                    config['public_ip_address']['properties'] = pip_dict['ip_configuration']
        self.log(result, pretty_print=True)
        if self.state != 'absent' and (not result['powerstate']):
            self.fail('Failed to determine PowerState of virtual machine {0}'.format(self.name))
        return result

    def power_off_vm(self):
        self.log('Powered off virtual machine {0} - Skip_Shutdown {1}'.format(self.name, self.force))
        self.results['actions'].append('Powered off virtual machine {0} - Skip_Shutdown {1}'.format(self.name, self.force))
        try:
            poller = self.compute_client.virtual_machines.begin_power_off(self.resource_group, self.name, skip_shutdown=self.force)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error powering off virtual machine {0} - {1}'.format(self.name, str(exc)))
        return True

    def power_on_vm(self):
        self.results['actions'].append('Powered on virtual machine {0}'.format(self.name))
        self.log('Power on virtual machine {0}'.format(self.name))
        try:
            poller = self.compute_client.virtual_machines.begin_start(self.resource_group, self.name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error powering on virtual machine {0} - {1}'.format(self.name, str(exc)))
        return True

    def restart_vm(self):
        self.results['actions'].append('Restarted virtual machine {0}'.format(self.name))
        self.log('Restart virtual machine {0}'.format(self.name))
        try:
            poller = self.compute_client.virtual_machines.begin_restart(self.resource_group, self.name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error restarting virtual machine {0} - {1}'.format(self.name, str(exc)))
        return True

    def deallocate_vm(self):
        self.results['actions'].append('Deallocated virtual machine {0}'.format(self.name))
        self.log('Deallocate virtual machine {0}'.format(self.name))
        try:
            poller = self.compute_client.virtual_machines.begin_deallocate(self.resource_group, self.name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deallocating virtual machine {0} - {1}'.format(self.name, str(exc)))
        return True

    def generalize_vm(self):
        self.results['actions'].append('Generalize virtual machine {0}'.format(self.name))
        self.log('Generalize virtual machine {0}'.format(self.name))
        try:
            response = self.compute_client.virtual_machines.generalize(self.resource_group, self.name)
            if isinstance(response, LROPoller):
                self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error generalizing virtual machine {0} - {1}'.format(self.name, str(exc)))
        return True

    def remove_autocreated_resources(self, tags):
        if tags:
            sa_name = tags.get('_own_sa_')
            nic_name = tags.get('_own_nic_')
            pip_name = tags.get('_own_pip_')
            nsg_name = tags.get('_own_nsg_')
            if sa_name:
                self.delete_storage_account(self.resource_group, sa_name)
            if nic_name:
                self.delete_nic(self.resource_group, nic_name)
            if pip_name:
                self.delete_pip(self.resource_group, pip_name)
            if nsg_name:
                self.delete_nsg(self.resource_group, nsg_name)

    def delete_vm(self, vm):
        vhd_uris = []
        managed_disk_ids = []
        nic_names = []
        pip_names = []
        if 'all_autocreated' not in self.remove_on_absent:
            if self.remove_on_absent.intersection(set(['all', 'virtual_storage'])):
                if vm.storage_profile.os_disk.managed_disk:
                    self.log('Storing managed disk ID for deletion')
                    managed_disk_ids.append(vm.storage_profile.os_disk.managed_disk.id)
                elif vm.storage_profile.os_disk.vhd:
                    self.log('Storing VHD URI for deletion')
                    vhd_uris.append(vm.storage_profile.os_disk.vhd.uri)
                data_disks = vm.storage_profile.data_disks
                for data_disk in data_disks:
                    if data_disk is not None:
                        if data_disk.vhd:
                            vhd_uris.append(data_disk.vhd.uri)
                        elif data_disk.managed_disk:
                            managed_disk_ids.append(data_disk.managed_disk.id)
                self.log('VHD URIs to delete: {0}'.format(', '.join(vhd_uris)))
                self.results['deleted_vhd_uris'] = vhd_uris
                self.log('Managed disk IDs to delete: {0}'.format(', '.join(managed_disk_ids)))
                self.results['deleted_managed_disk_ids'] = managed_disk_ids
            if self.remove_on_absent.intersection(set(['all', 'network_interfaces'])):
                self.log('Storing NIC names for deletion.')
                for interface in vm.network_profile.network_interfaces:
                    id_dict = azure_id_to_dict(interface.id)
                    nic_names.append(dict(name=id_dict['networkInterfaces'], resource_group=id_dict['resourceGroups']))
                self.log('NIC names to delete {0}'.format(str(nic_names)))
                self.results['deleted_network_interfaces'] = nic_names
                if self.remove_on_absent.intersection(set(['all', 'public_ips'])):
                    for nic_dict in nic_names:
                        nic = self.get_network_interface(nic_dict['resource_group'], nic_dict['name'])
                        for ipc in nic.ip_configurations:
                            if ipc.public_ip_address:
                                pip_dict = azure_id_to_dict(ipc.public_ip_address.id)
                                pip_names.append(dict(name=pip_dict['publicIPAddresses'], resource_group=pip_dict['resourceGroups']))
                    self.log('Public IPs to  delete are {0}'.format(str(pip_names)))
                    self.results['deleted_public_ips'] = pip_names
        self.log('Deleting virtual machine {0}'.format(self.name))
        self.results['actions'].append('Deleted virtual machine {0}'.format(self.name))
        try:
            poller = self.compute_client.virtual_machines.begin_delete(self.resource_group, self.name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting virtual machine {0} - {1}'.format(self.name, str(exc)))
        if self.remove_on_absent.intersection(set(['all', 'virtual_storage'])):
            self.log('Deleting VHDs')
            self.delete_vm_storage(vhd_uris)
            self.log('Deleting managed disks')
            self.delete_managed_disks(managed_disk_ids)
        if 'all' in self.remove_on_absent or 'all_autocreated' in self.remove_on_absent:
            self.remove_autocreated_resources(vm.tags)
        if self.remove_on_absent.intersection(set(['all', 'network_interfaces'])):
            self.log('Deleting network interfaces')
            for nic_dict in nic_names:
                self.delete_nic(nic_dict['resource_group'], nic_dict['name'])
        if self.remove_on_absent.intersection(set(['all', 'public_ips'])):
            self.log('Deleting public IPs')
            for pip_dict in pip_names:
                self.delete_pip(pip_dict['resource_group'], pip_dict['name'])
        if 'all' in self.remove_on_absent or 'all_autocreated' in self.remove_on_absent:
            self.remove_autocreated_resources(vm.tags)
        return True

    def get_network_interface(self, resource_group, name):
        try:
            nic = self.network_client.network_interfaces.get(resource_group, name)
            return nic
        except ResourceNotFoundError as exc:
            self.fail('Error fetching network interface {0} - {1}'.format(name, str(exc)))
        return True

    def delete_nic(self, resource_group, name):
        self.log('Deleting network interface {0}'.format(name))
        self.results['actions'].append('Deleted network interface {0}'.format(name))
        try:
            poller = self.network_client.network_interfaces.begin_delete(resource_group, name)
        except Exception as exc:
            self.fail('Error deleting network interface {0} - {1}'.format(name, str(exc)))
        self.get_poller_result(poller)
        return True

    def delete_pip(self, resource_group, name):
        self.results['actions'].append('Deleted public IP {0}'.format(name))
        try:
            poller = self.network_client.public_ip_addresses.begin_delete(resource_group, name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting {0} - {1}'.format(name, str(exc)))
        return True

    def delete_nsg(self, resource_group, name):
        self.results['actions'].append('Deleted NSG {0}'.format(name))
        try:
            poller = self.network_client.network_security_groups.begin_delete(resource_group, name)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error deleting {0} - {1}'.format(name, str(exc)))
        return True

    def delete_managed_disks(self, managed_disk_ids):
        for mdi in managed_disk_ids:
            try:
                poller = self.rm_client.resources.begin_delete_by_id(mdi, '2017-03-30')
                self.get_poller_result(poller)
            except Exception as exc:
                self.fail('Error deleting managed disk {0} - {1}'.format(mdi, str(exc)))
        return True

    def delete_storage_account(self, resource_group, name):
        self.log('Delete storage account {0}'.format(name))
        self.results['actions'].append('Deleted storage account {0}'.format(name))
        try:
            self.storage_client.storage_accounts.delete(self.resource_group, name)
        except Exception as exc:
            self.fail('Error deleting storage account {0} - {1}'.format(name, str(exc)))
        return True

    def delete_vm_storage(self, vhd_uris):
        for uri in vhd_uris:
            self.log("Extracting info from blob uri '{0}'".format(uri))
            try:
                blob_parts = extract_names_from_blob_uri(uri, self._cloud_environment.suffixes.storage_endpoint)
            except Exception as exc:
                self.fail('Error parsing blob URI {0}'.format(str(exc)))
            storage_account_name = blob_parts['accountname']
            container_name = blob_parts['containername']
            blob_name = blob_parts['blobname']
            blob_service_client = self.get_blob_service_client(self.resource_group, storage_account_name)
            self.log('Delete blob {0}:{1}'.format(container_name, blob_name))
            self.results['actions'].append('Deleted blob {0}:{1}'.format(container_name, blob_name))
            try:
                blob_service_client.get_blob_client(container=container_name, blob=blob_name).delete_blob()
            except Exception as exc:
                self.fail('Error deleting blob {0}:{1} - {2}'.format(container_name, blob_name, str(exc)))
        return True

    def get_marketplace_image_version(self):
        try:
            versions = self.compute_client.virtual_machine_images.list(self.location, self.image['publisher'], self.image['offer'], self.image['sku'], orderby='name')
        except Exception as exc:
            self.fail('Error fetching image {0} {1} {2} - {3}'.format(self.image['publisher'], self.image['offer'], self.image['sku'], str(exc)))
        if versions and len(versions) > 0:
            if self.image['version'] == 'latest':
                return versions[len(versions) - 1]
            for version in versions:
                if version.name == self.image['version']:
                    return version
        self.fail('Error could not find image {0} {1} {2} {3}'.format(self.image['publisher'], self.image['offer'], self.image['sku'], self.image['version']))
        return None

    def get_custom_image_reference(self, name, resource_group=None):
        try:
            if resource_group:
                vm_images = self.compute_client.images.list_by_resource_group(resource_group)
            else:
                vm_images = self.compute_client.images.list()
        except Exception as exc:
            self.fail('Error fetching custom images from subscription - {0}'.format(str(exc)))
        for vm_image in vm_images:
            if vm_image.name == name:
                self.log('Using custom image id {0}'.format(vm_image.id))
                return self.compute_models.ImageReference(id=vm_image.id)
        self.fail('Error could not find image with name {0}'.format(name))
        return None

    def get_proximity_placement_group(self, resource_group, name):
        try:
            return self.compute_client.proximity_placement_groups.get(resource_group, name)
        except Exception as exc:
            self.fail('Error fetching proximity placement group {0} - {1}'.format(name, str(exc)))

    def get_availability_set(self, resource_group, name):
        try:
            return self.compute_client.availability_sets.get(resource_group, name)
        except Exception as exc:
            self.fail('Error fetching availability set {0} - {1}'.format(name, str(exc)))

    def get_storage_account(self, resource_group, name):
        try:
            account = self.storage_client.storage_accounts.get_properties(resource_group, name)
            return account
        except Exception as exc:
            self.fail('Error fetching storage account {0} - {1}'.format(name, str(exc)))

    def create_or_update_vm(self, params, remove_autocreated_on_failure):
        try:
            poller = self.compute_client.virtual_machines.begin_create_or_update(self.resource_group, self.name, params)
            self.get_poller_result(poller)
        except Exception as exc:
            if remove_autocreated_on_failure:
                self.remove_autocreated_resources(params.tags)
            self.fail('Error creating or updating virtual machine {0} - {1}'.format(self.name, str(exc)))

    def vm_size_is_valid(self):
        """
        Validate self.vm_size against the list of virtual machine sizes available for the account and location.

        :return: boolean
        """
        try:
            sizes = self.compute_client.virtual_machine_sizes.list(self.location)
        except Exception as exc:
            self.fail('Error retrieving available machine sizes - {0}'.format(str(exc)))
        for size in sizes:
            if size.name == self.vm_size:
                return True
        return False

    def create_default_storage_account(self, vm_dict=None):
        """
        Create (once) a default storage account <vm name>XXXX, where XXXX is a random number.
        NOTE: If <vm name>XXXX exists, use it instead of failing.  Highly unlikely.
        If this method is called multiple times across executions it will return the same
        storage account created with the random name which is stored in a tag on the VM.

        vm_dict is passed in during an update, so we can obtain the _own_sa_ tag and return
        the default storage account we created in a previous invocation

        :return: storage account object
        """
        account = None
        valid_name = False
        if self.tags is None:
            self.tags = {}
        if self.tags.get('_own_sa_', None):
            return self.get_storage_account(self.resource_group, self.tags['_own_sa_'])
        if vm_dict and vm_dict.get('tags', {}).get('_own_sa_', None):
            return self.get_storage_account(self.resource_group, vm_dict['tags']['_own_sa_'])
        storage_account_name_base = re.sub('[^a-zA-Z0-9]', '', self.name[:20].lower())
        for i in range(0, 5):
            rand = random.randrange(1000, 9999)
            storage_account_name = storage_account_name_base + str(rand)
            if self.check_storage_account_name(storage_account_name):
                valid_name = True
                break
        if not valid_name:
            self.fail('Failed to create a unique storage account name for {0}. Try using a different VM name.'.format(self.name))
        try:
            account = self.storage_client.storage_accounts.get_properties(self.resource_group, storage_account_name)
        except Exception:
            pass
        if account:
            self.log('Storage account {0} found.'.format(storage_account_name))
            self.check_provisioning_state(account)
            return account
        sku = self.storage_models.Sku(name=self.storage_models.SkuName.standard_lrs)
        sku.tier = self.storage_models.SkuTier.standard
        kind = self.storage_models.Kind.storage
        parameters = self.storage_models.StorageAccountCreateParameters(sku=sku, kind=kind, location=self.location)
        self.log('Creating storage account {0} in location {1}'.format(storage_account_name, self.location))
        self.results['actions'].append('Created storage account {0}'.format(storage_account_name))
        try:
            poller = self.storage_client.storage_accounts.begin_create(self.resource_group, storage_account_name, parameters)
            self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Failed to create storage account: {0} - {1}'.format(storage_account_name, str(exc)))
        self.tags['_own_sa_'] = storage_account_name
        return self.get_storage_account(self.resource_group, storage_account_name)

    def check_storage_account_name(self, name):
        self.log('Checking storage account name availability for {0}'.format(name))
        try:
            account_name = self.storage_models.StorageAccountCheckNameAvailabilityParameters(name=name)
            response = self.storage_client.storage_accounts.check_name_availability(account_name)
            if response.reason == 'AccountNameInvalid':
                raise Exception('Invalid default storage account name: {0}'.format(name))
        except Exception as exc:
            self.fail('Error checking storage account name availability for {0} - {1}'.format(name, str(exc)))
        return response.name_available

    def create_default_nic(self):
        """
        Create a default Network Interface <vm name>01. Requires an existing virtual network
        with one subnet. If NIC <vm name>01 exists, use it. Otherwise, create one.

        :return: NIC object
        """
        network_interface_name = self.name + '01'
        nic = None
        if self.tags is None:
            self.tags = {}
        self.log('Create default NIC {0}'.format(network_interface_name))
        self.log('Check to see if NIC {0} exists'.format(network_interface_name))
        try:
            nic = self.network_client.network_interfaces.get(self.resource_group, network_interface_name)
        except ResourceNotFoundError:
            pass
        if nic:
            self.log('NIC {0} found.'.format(network_interface_name))
            self.check_provisioning_state(nic)
            return nic
        self.log('NIC {0} does not exist.'.format(network_interface_name))
        virtual_network_resource_group = None
        if self.virtual_network_resource_group:
            virtual_network_resource_group = self.virtual_network_resource_group
        else:
            virtual_network_resource_group = self.resource_group
        if self.virtual_network_name:
            try:
                self.network_client.virtual_networks.get(virtual_network_resource_group, self.virtual_network_name)
                virtual_network_name = self.virtual_network_name
            except ResourceNotFoundError as exc:
                self.fail('Error: fetching virtual network {0} - {1}'.format(self.virtual_network_name, str(exc)))
        else:
            no_vnets_msg = 'Error: unable to find virtual network in resource group {0}. A virtual network with at least one subnet must exist in order to create a NIC for the virtual machine.'.format(virtual_network_resource_group)
            virtual_network_name = None
            try:
                vnets = self.network_client.virtual_networks.list(virtual_network_resource_group)
            except ResourceNotFoundError:
                self.log('cloud error!')
                self.fail(no_vnets_msg)
            for vnet in vnets:
                virtual_network_name = vnet.name
                self.log('vnet name: {0}'.format(vnet.name))
                break
            if not virtual_network_name:
                self.fail(no_vnets_msg)
        if self.subnet_name:
            try:
                subnet = self.network_client.subnets.get(virtual_network_resource_group, virtual_network_name, self.subnet_name)
                subnet_id = subnet.id
            except Exception as exc:
                self.fail('Error: fetching subnet {0} - {1}'.format(self.subnet_name, str(exc)))
        else:
            no_subnets_msg = 'Error: unable to find a subnet in virtual network {0}. A virtual network with at least one subnet must exist in order to create a NIC for the virtual machine.'.format(virtual_network_name)
            subnet_id = None
            try:
                subnets = self.network_client.subnets.list(virtual_network_resource_group, virtual_network_name)
            except Exception:
                self.fail(no_subnets_msg)
            for subnet in subnets:
                subnet_id = subnet.id
                self.log('subnet id: {0}'.format(subnet_id))
                break
            if not subnet_id:
                self.fail(no_subnets_msg)
        pip = None
        if self.public_ip_allocation_method != 'Disabled':
            self.results['actions'].append('Created default public IP {0}'.format(self.name + '01'))
            sku = self.network_models.PublicIPAddressSku(name='Standard') if self.zones else None
            pip_facts = self.create_default_pip(self.resource_group, self.location, self.name + '01', self.public_ip_allocation_method, sku=sku)
            pip = self.network_models.PublicIPAddress(id=pip_facts.id, location=pip_facts.location, resource_guid=pip_facts.resource_guid, sku=sku)
            self.tags['_own_pip_'] = self.name + '01'
        self.tags['_own_nsg_'] = self.name + '01'
        parameters = self.network_models.NetworkInterface(location=self.location, ip_configurations=[self.network_models.NetworkInterfaceIPConfiguration(private_ip_allocation_method='Dynamic')])
        parameters.ip_configurations[0].subnet = self.network_models.Subnet(id=subnet_id)
        parameters.ip_configurations[0].name = 'default'
        if self.created_nsg:
            self.results['actions'].append('Created default security group {0}'.format(self.name + '01'))
            group = self.create_default_securitygroup(self.resource_group, self.location, self.name + '01', self.os_type, self.open_ports)
            parameters.network_security_group = self.network_models.NetworkSecurityGroup(id=group.id, location=group.location, resource_guid=group.resource_guid)
        parameters.ip_configurations[0].public_ip_address = pip
        self.log('Creating NIC {0}'.format(network_interface_name))
        self.log(self.serialize_obj(parameters, 'NetworkInterface'), pretty_print=True)
        self.results['actions'].append('Created NIC {0}'.format(network_interface_name))
        try:
            poller = self.network_client.network_interfaces.begin_create_or_update(self.resource_group, network_interface_name, parameters)
            new_nic = self.get_poller_result(poller)
            self.tags['_own_nic_'] = network_interface_name
        except Exception as exc:
            self.fail('Error creating network interface {0} - {1}'.format(network_interface_name, str(exc)))
        return new_nic

    def parse_network_interface(self, nic):
        nic = self.parse_resource_to_dict(nic)
        if 'name' not in nic:
            self.fail('Invalid network interface {0}'.format(str(nic)))
        return format_resource_id(val=nic['name'], subscription_id=nic['subscription_id'], resource_group=nic['resource_group'], namespace='Microsoft.Network', types='networkInterfaces')