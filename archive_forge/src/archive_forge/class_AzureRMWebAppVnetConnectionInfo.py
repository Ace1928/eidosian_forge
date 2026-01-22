from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMWebAppVnetConnectionInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), resource_group=dict(type='str', required=True))
        self.results = dict(changed=False, connection=dict())
        self.name = None
        self.resource_group = None
        super(AzureRMWebAppVnetConnectionInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        vnet = self.get_vnet_connection()
        if vnet:
            self.results['connection'] = self.set_results(vnet)
        return self.results

    def get_vnet_connection(self):
        connections = self.list_vnet_connections()
        for connection in connections:
            if connection.is_swift:
                return connection
        return None

    def list_vnet_connections(self):
        try:
            return self.web_client.web_apps.list_vnet_connections(resource_group_name=self.resource_group, name=self.name)
        except Exception as exc:
            self.fail('Error getting webapp vnet connections {0} (rg={1}) - {2}'.format(self.name, self.resource_group, str(exc)))

    def set_results(self, vnet):
        vnet_dict = vnet.as_dict()
        output = dict()
        output['id'] = vnet_dict['id']
        output['name'] = vnet_dict['name']
        subnet_id = vnet_dict['vnet_resource_id']
        output['vnet_resource_id'] = subnet_id
        subnet_detail = self.get_subnet_detail(subnet_id)
        output['vnet_resource_group'] = subnet_detail['resource_group']
        output['vnet_name'] = subnet_detail['vnet_name']
        output['subnet_name'] = subnet_detail['subnet_name']
        return output