from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMVirtualHubConnectionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), virtual_hub_name=dict(type='str', required=True))
        self.resource_group = None
        self.name = None
        self.virtual_hub_name = None
        self.results = dict(changed=False)
        self.state = None
        self.status_code = [200]
        super(AzureRMVirtualHubConnectionInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name is not None:
            self.results['virtual_hub_connection'] = self.format_item(self.get())
        else:
            self.results['virtual_hub_connection'] = self.format_item(self.list())
        return self.results

    def get(self):
        response = None
        try:
            response = self.network_client.hub_virtual_network_connections.get(resource_group_name=self.resource_group, virtual_hub_name=self.virtual_hub_name, connection_name=self.name)
        except ResourceNotFoundError:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def list(self):
        response = None
        try:
            response = self.network_client.hub_virtual_network_connections.list(resource_group_name=self.resource_group, virtual_hub_name=self.virtual_hub_name)
        except Exception:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def format_item(self, item):
        if item is None:
            return None
        elif hasattr(item, 'as_dict'):
            return [item.as_dict()]
        else:
            result = []
            items = list(item)
            for tmp in items:
                result.append(tmp.as_dict())
            return result