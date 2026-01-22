from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMVirtualMachineScaleSetExtensionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), vmss_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.vmss_name = None
        self.name = None
        super(AzureRMVirtualMachineScaleSetExtensionInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_virtualmachinescalesetextension_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_virtualmachinescalesetextension_facts' module has been renamed to" + " 'azure_rm_virtualmachinescalesetextension_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['extensions'] = self.get_extensions()
        else:
            self.results['extensions'] = self.list_extensions()
        return self.results

    def get_extensions(self):
        response = None
        results = []
        try:
            response = self.compute_client.virtual_machine_scale_set_extensions.get(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name, vmss_extension_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Virtual Machine Extension.')
        if response:
            results.append(self.format_response(response))
        return results

    def list_extensions(self):
        response = None
        results = []
        try:
            response = self.compute_client.virtual_machine_scale_set_extensions.list(resource_group_name=self.resource_group, vm_scale_set_name=self.vmss_name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Virtual Machine Extension.')
        if response is not None:
            for item in response:
                results.append(self.format_response(item))
        return results

    def format_response(self, item):
        id_template = '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.Compute/virtualMachineScaleSets/{2}/extensions/{3}'
        d = item.as_dict()
        d = {'id': id_template.format(self.subscription_id, self.resource_group, self.vmss_name, d.get('name')), 'resource_group': self.resource_group, 'vmss_name': self.vmss_name, 'name': d.get('name'), 'publisher': d.get('publisher'), 'type': d.get('type'), 'settings': d.get('settings'), 'auto_upgrade_minor_version': d.get('auto_upgrade_minor_version'), 'provisioning_state': d.get('provisioning_state')}
        return d