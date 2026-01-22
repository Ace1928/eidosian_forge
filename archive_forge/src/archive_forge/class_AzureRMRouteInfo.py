from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBase
class AzureRMRouteInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), route_table_name=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.resource_group = None
        self.route_table_name = None
        self.name = None
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.url = None
        self.status_code = [200]
        self.tags = None
        self.mgmt_client = None
        super(AzureRMRouteInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.resource_group is not None and self.route_table_name is not None and (self.name is not None):
            self.results['routes'] = self.format_item(self.get())
        elif self.resource_group is not None and self.route_table_name is not None:
            self.results['routes'] = self.format_item(self.list())
        return self.results

    def get(self):
        response = None
        try:
            response = self.network_client.routes.get(resource_group_name=self.resource_group, route_table_name=self.route_table_name, route_name=self.name)
        except ResourceNotFoundError as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def list(self):
        response = None
        try:
            response = self.network_client.routes.list(resource_group_name=self.resource_group, route_table_name=self.route_table_name)
        except ResourceNotFoundError as e:
            self.log('Could not get info for @(Model.ModuleOperationNameUpper).')
        return response

    def format_item(self, item):
        if hasattr(item, 'as_dict'):
            return [item.as_dict()]
        else:
            result = []
            items = list(item)
            for tmp in items:
                result.append(tmp.as_dict())
            return result