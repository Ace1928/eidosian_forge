from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMVpnSiteLinkInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), vpn_site_name=dict(type='str', required=True), name=dict(type='str'))
        self.resource_group = None
        self.vpn_site_name = None
        name = None
        self.results = dict(changed=False)
        self.state = None
        super(AzureRMVpnSiteLinkInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.resource_group is not None and self.vpn_site_name is not None and (self.name is not None):
            self.results['vpn_site_links'] = self.format_item(self.get())
        elif self.resource_group is not None and self.vpn_site_name is not None:
            self.results['vpn_site_links'] = self.format_item(self.list_by_vpn_site())
        return self.results

    def get(self):
        response = None
        try:
            response = self.network_client.vpn_site_links.get(resource_group_name=self.resource_group, vpn_site_name=self.vpn_site_name, vpn_site_link_name=self.name)
        except ResourceNotFoundError as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def list_by_vpn_site(self):
        response = None
        try:
            response = self.network_client.vpn_site_links.list_by_vpn_site(resource_group_name=self.resource_group, vpn_site_name=self.vpn_site_name)
        except ResourceNotFoundError as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def format_item(self, item):
        if hasattr(item, 'as_dict'):
            return [item.as_dict()]
        elif item is not None:
            result = []
            items = list(item)
            for tmp in items:
                result.append(tmp.as_dict())
            return result
        else:
            return None