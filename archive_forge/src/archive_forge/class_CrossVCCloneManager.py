from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
class CrossVCCloneManager(PyVmomi):

    def __init__(self, module):
        super(CrossVCCloneManager, self).__init__(module)
        self.config_spec = vim.vm.ConfigSpec()
        self.clone_spec = vim.vm.CloneSpec()
        self.relocate_spec = vim.vm.RelocateSpec()
        self.service_locator = vim.ServiceLocator()
        self.destination_vcenter = self.params['destination_vcenter']
        self.destination_vcenter_username = self.params['destination_vcenter_username']
        self.destination_vcenter_password = self.params['destination_vcenter_password']
        self.destination_vcenter_port = self.params.get('port', 443)
        self.destination_vcenter_validate_certs = self.params.get('destination_vcenter_validate_certs', None)
        self.timeout = self.params.get('timeout')

    def get_new_vm_info(self, vm):
        self.destination_content = connect_to_api(self.module, hostname=self.destination_vcenter, username=self.destination_vcenter_username, password=self.destination_vcenter_password, port=self.destination_vcenter_port, validate_certs=self.destination_vcenter_validate_certs)
        info = {}
        vm_obj = find_vm_by_name(content=self.destination_content, vm_name=vm)
        if vm_obj is None:
            self.module.fail_json(msg='Newly cloned VM is not found in the destination VCenter')
        else:
            vm_facts = gather_vm_facts(self.destination_content, vm_obj)
            info['vm_name'] = vm
            info['vcenter'] = self.destination_vcenter
            info['host'] = vm_facts['hw_esxi_host']
            info['datastore'] = vm_facts['hw_datastores']
            info['vm_folder'] = vm_facts['hw_folder']
            info['power_on'] = vm_facts['hw_power_status']
        return info

    def clone(self):
        vm_folder = find_folder_by_name(content=self.destination_content, folder_name=self.params['destination_vm_folder'])
        if not vm_folder:
            self.module.fail_json(msg='Destination folder does not exist. Please refer to the documentation to correctly specify the folder.')
        vm_name = self.params['destination_vm_name']
        task = self.vm_obj.Clone(folder=vm_folder, name=vm_name, spec=self.clone_spec)
        wait_for_task(task, timeout=self.timeout)
        if task.info.state == 'error':
            result = {'changed': False, 'failed': True, 'msg': task.info.error.msg}
        else:
            vm_info = self.get_new_vm_info(vm_name)
            result = {'changed': True, 'failed': False, 'vm_info': vm_info}
        return result

    def sanitize_params(self):
        """
        this method is used to verify user provided parameters
        """
        self.vm_obj = self.get_vm()
        if self.vm_obj is None:
            vm_id = self.vm_uuid or self.vm_name or self.moid
            self.module.fail_json(msg='Failed to find the VM/template with %s' % vm_id)
        self.destination_content = connect_to_api(self.module, hostname=self.destination_vcenter, username=self.destination_vcenter_username, password=self.destination_vcenter_password, port=self.destination_vcenter_port, validate_certs=self.destination_vcenter_validate_certs)
        vm = find_vm_by_name(content=self.destination_content, vm_name=self.params['destination_vm_name'])
        if vm:
            self.module.exit_json(changed=False, msg='A VM with the given name already exists')
        datastore_name = self.params['destination_datastore']
        datastore_cluster = find_obj(self.destination_content, [vim.StoragePod], datastore_name)
        if datastore_cluster:
            datastore_name = self.get_recommended_datastore(datastore_cluster_obj=datastore_cluster)
        self.destination_datastore = find_datastore_by_name(content=self.destination_content, datastore_name=datastore_name)
        if self.destination_datastore is None:
            self.module.fail_json(msg='Destination datastore not found.')
        self.destination_host = find_hostsystem_by_name(content=self.destination_content, hostname=self.params['destination_host'])
        if self.destination_host is None:
            self.module.fail_json(msg='Destination host not found.')
        if self.params['destination_resource_pool']:
            self.destination_resource_pool = find_resource_pool_by_name(content=self.destination_content, resource_pool_name=self.params['destination_resource_pool'])
        else:
            self.destination_resource_pool = self.destination_host.parent.resourcePool

    def populate_specs(self):
        self.service_locator.instanceUuid = self.destination_content.about.instanceUuid
        self.service_locator.url = 'https://' + self.destination_vcenter + ':' + str(self.params['port']) + '/sdk'
        if not self.destination_vcenter_validate_certs:
            self.service_locator.sslThumbprint = self.get_cert_fingerprint(self.destination_vcenter, self.destination_vcenter_port, self.module.params['proxy_host'], self.module.params['proxy_port'])
        creds = vim.ServiceLocatorNamePassword()
        creds.username = self.destination_vcenter_username
        creds.password = self.destination_vcenter_password
        self.service_locator.credential = creds
        self.relocate_spec.datastore = self.destination_datastore
        self.relocate_spec.pool = self.destination_resource_pool
        self.relocate_spec.service = self.service_locator
        self.relocate_spec.host = self.destination_host
        self.clone_spec.config = self.config_spec
        self.clone_spec.powerOn = True if self.params['state'].lower() == 'poweredon' else False
        self.clone_spec.location = self.relocate_spec
        self.clone_spec.template = self.params['is_template']