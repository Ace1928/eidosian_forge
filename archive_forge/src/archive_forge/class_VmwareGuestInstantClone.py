from __future__ import (absolute_import, division, print_function)
import time
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class VmwareGuestInstantClone(PyVmomi):

    def __init__(self, module):
        """Constructor."""
        super().__init__(module)
        self.instant_clone_spec = vim.vm.InstantCloneSpec()
        self.relocate_spec = vim.vm.RelocateSpec()
        self.vm_name = self.params.get('name')
        self.parent_vm = self.params.get('parent_vm')
        self.datacenter = self.params.get('datacenter')
        self.datastore = self.params.get('datastore')
        self.hostname = self.params.get('hostname')
        self.folder = self.params.get('folder')
        self.resource_pool = self.params.get('resource_pool')
        self.host = self.params.get('host')
        self.username = self.params.get('username')
        self.password = self.params.get('password')
        self.validate_certs = self.params.get('validate_certs')
        self.moid = self.params.get('moid')
        self.uuid = self.params.get('uuid')
        self.port = self.params.get('port')
        self.use_instance_uuid = self.params.get('use_instance_uuid')
        self.wait_vm_tools = self.params.get('wait_vm_tools')
        self.wait_vm_tools_timeout = self.params.get('wait_vm_tools_timeout')
        self.guestinfo_vars = self.params.get('guestinfo_vars')

    def get_new_vm_info(self, vm):
        info = {}
        vm_obj = find_vm_by_name(content=self.destination_content, vm_name=vm)
        if vm_obj is None:
            self.module.fail_json(msg='Newly Instant cloned VM is not found in the VCenter')
        vm_facts = self.gather_facts(vm_obj)
        info['vm_name'] = vm
        info['vcenter'] = self.hostname
        info['host'] = vm_facts['hw_esxi_host']
        info['datastore'] = vm_facts['hw_datastores']
        info['vm_folder'] = vm_facts['hw_folder']
        info['instance_uuid'] = vm_facts['instance_uuid']
        return info

    def Instant_clone(self):
        if self.vm_obj is None:
            vm_id = self.parent_vm or self.uuid or self.moid
            self.module.fail_json(msg='Failed to find the VM/template with %s' % vm_id)
        try:
            task = self.vm_obj.InstantClone_Task(spec=self.instant_clone_spec)
            wait_for_task(task)
            vm_info = self.get_new_vm_info(self.vm_name)
            result = {'changed': True, 'failed': False, 'vm_info': vm_info}
        except TaskError as task_e:
            self.module.fail_json(msg=to_native(task_e))
        self.destination_content = connect_to_api(self.module, hostname=self.hostname, username=self.username, password=self.password, port=self.port, validate_certs=self.validate_certs)
        vm_IC = find_vm_by_name(content=self.destination_content, vm_name=self.params['name'])
        if vm_IC and self.params.get('guestinfo_vars'):
            guest_custom_mng = self.destination_content.guestCustomizationManager
            auth_obj = vim.vm.guest.NamePasswordAuthentication()
            guest_user = self.params.get('vm_username')
            guest_password = self.params.get('vm_password')
            auth_obj.username = guest_user
            auth_obj.password = guest_password
            guestinfo_vars = self.params.get('guestinfo_vars')
            customization_spec = vim.vm.customization.Specification()
            customization_spec.globalIPSettings = vim.vm.customization.GlobalIPSettings()
            customization_spec.globalIPSettings.dnsServerList = [guestinfo_vars[0]['dns']]
            customization_spec.identity = vim.vm.customization.LinuxPrep()
            customization_spec.identity.domain = guestinfo_vars[0]['domain']
            customization_spec.identity.hostName = vim.vm.customization.FixedName()
            customization_spec.identity.hostName.name = guestinfo_vars[0]['hostname']
            customization_spec.nicSettingMap = []
            adapter_mapping_obj = vim.vm.customization.AdapterMapping()
            adapter_mapping_obj.adapter = vim.vm.customization.IPSettings()
            adapter_mapping_obj.adapter.ip = vim.vm.customization.FixedIp()
            adapter_mapping_obj.adapter.ip.ipAddress = guestinfo_vars[0]['ipaddress']
            adapter_mapping_obj.adapter.subnetMask = guestinfo_vars[0]['netmask']
            adapter_mapping_obj.adapter.gateway = [guestinfo_vars[0]['gateway']]
            customization_spec.nicSettingMap.append(adapter_mapping_obj)
            try:
                task_guest = guest_custom_mng.CustomizeGuest_Task(vm_IC, auth_obj, customization_spec)
                wait_for_task(task_guest)
                vm_info = self.get_new_vm_info(self.vm_name)
                result = {'changed': True, 'failed': False, 'vm_info': vm_info}
            except TaskError as task_e:
                self.module.fail_json(msg=to_native(task_e))
            instant_vm_obj = find_vm_by_id(content=self.content, vm_id=vm_info['instance_uuid'], vm_id_type='instance_uuid')
            set_vm_power_state(content=self.content, vm=instant_vm_obj, state='rebootguest', force=False)
            if self.wait_vm_tools:
                interval = 15
                while self.wait_vm_tools_timeout > 0:
                    if instant_vm_obj.guest.toolsRunningStatus != 'guestToolsRunning':
                        break
                    self.wait_vm_tools_timeout -= interval
                    time.sleep(interval)
                while self.wait_vm_tools_timeout > 0:
                    if instant_vm_obj.guest.toolsRunningStatus == 'guestToolsRunning':
                        break
                    self.wait_vm_tools_timeout -= interval
                    time.sleep(interval)
                if self.wait_vm_tools_timeout <= 0:
                    self.module.fail_json(msg='Timeout has been reached for waiting to start the vm tools.')
        return result

    def sanitize_params(self):
        """
        Verify user-provided parameters
        """
        self.destination_content = connect_to_api(self.module, hostname=self.hostname, username=self.username, password=self.password, port=self.port, validate_certs=self.validate_certs)
        use_instance_uuid = self.params.get('use_instance_uuid') or False
        if 'parent_vm' in self.params and self.params['parent_vm']:
            self.vm_obj = find_vm_by_name(content=self.destination_content, vm_name=self.parent_vm)
        elif 'uuid' in self.params and self.params['uuid']:
            if not use_instance_uuid:
                self.vm_obj = find_vm_by_id(content=self.destination_content, vm_id=self.params['uuid'], vm_id_type='uuid')
            elif use_instance_uuid:
                self.vm_obj = find_vm_by_id(content=self.destination_content, vm_id=self.params['uuid'], vm_id_type='instance_uuid')
        elif 'moid' in self.params and self.params['moid']:
            self.vm_obj = vim.VirtualMachine(self.params['moid'], self.si._stub)
        if self.vm_obj is None:
            vm_id = self.parent_vm or self.uuid or self.moid
            self.module.fail_json(msg='Failed to find the VM/template with %s' % vm_id)
        vm = find_vm_by_name(content=self.destination_content, vm_name=self.params['name'])
        if vm:
            self.module.exit_json(changed=False, msg='A VM with the given name already exists')
        self.datacenter = self.find_datacenter_by_name(self.params['datacenter'])
        if self.datacenter is None:
            self.module.fail_json(msg='Datacenter not found.')
        datastore_name = self.params['datastore']
        datastore_cluster = find_obj(self.destination_content, [vim.StoragePod], datastore_name)
        if datastore_cluster:
            datastore_name = self.get_recommended_datastore(datastore_cluster_obj=datastore_cluster)
        self.datastore = self.find_datastore_by_name(datastore_name=datastore_name)
        if self.datastore is None:
            self.module.fail_json(msg='Datastore not found.')
        if self.params['folder']:
            self.folder = self.find_folder_by_fqpn(folder_name=self.params['folder'], datacenter_name=self.params['datacenter'], folder_type='vm')
            if self.folder is None:
                self.module.fail_json(msg='Folder not found.')
        else:
            self.folder = self.datacenter.vmFolder
        self.host = self.find_hostsystem_by_name(host_name=self.params['host'])
        if self.host is None:
            self.module.fail_json(msg='Host not found.')
        if self.params['resource_pool']:
            self.resource_pool = self.find_resource_pool_by_name(resource_pool_name=self.params['resource_pool'])
            if self.resource_pool is None:
                self.module.fail_json(msg='Resource Pool not found.')
        else:
            self.resource_pool = self.host.parent.resourcePool
        if self.params['guestinfo_vars']:
            self.guestinfo_vars = self.dict_to_optionvalues()
        else:
            self.guestinfo_vars = None

    def dict_to_optionvalues(self):
        optionvalues = []
        for dictionary in self.params['guestinfo_vars']:
            for key, value in dictionary.items():
                opt = vim.option.OptionValue()
                opt.key, opt.value = ('guestinfo.ic.' + key, value)
                optionvalues.append(opt)
        return optionvalues

    def populate_specs(self):
        self.relocate_spec.datastore = self.datastore
        self.relocate_spec.pool = self.resource_pool
        self.relocate_spec.folder = self.folder
        self.instant_clone_spec.name = self.vm_name
        self.instant_clone_spec.location = self.relocate_spec
        self.instant_clone_spec.config = self.guestinfo_vars