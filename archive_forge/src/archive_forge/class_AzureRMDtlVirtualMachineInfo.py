from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDtlVirtualMachineInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.lab_name = None
        self.name = None
        self.tags = None
        super(AzureRMDtlVirtualMachineInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_devtestlabvirtualmachine_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_devtestlabvirtualmachine_facts' module has been renamed to 'azure_rm_devtestlabvirtualmachine_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name:
            self.results['virtualmachines'] = self.get()
        else:
            self.results['virtualmachines'] = self.list()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.virtual_machines.get(resource_group_name=self.resource_group, lab_name=self.lab_name, name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get facts for Virtual Machine.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def list(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.virtual_machines.list(resource_group_name=self.resource_group, lab_name=self.lab_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.fail('Could not get facts for Virtual Machine.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'id': d.get('id', None), 'resource_group': self.parse_resource_to_dict(d.get('id')).get('resource_group'), 'lab_name': self.parse_resource_to_dict(d.get('id')).get('name'), 'name': d.get('name'), 'notes': d.get('notes'), 'disallow_public_ip_address': d.get('disallow_public_ip_address'), 'expiration_date': d.get('expiration_date'), 'image': d.get('gallery_image_reference'), 'os_type': d.get('os_type').lower(), 'vm_size': d.get('size'), 'user_name': d.get('user_name'), 'storage_type': d.get('storage_type').lower(), 'compute_vm_id': d.get('compute_id'), 'compute_vm_resource_group': self.parse_resource_to_dict(d.get('compute_id')).get('resource_group'), 'compute_vm_name': self.parse_resource_to_dict(d.get('compute_id')).get('name'), 'fqdn': d.get('fqdn'), 'provisioning_state': d.get('provisioning_state'), 'tags': d.get('tags', None)}
        return d