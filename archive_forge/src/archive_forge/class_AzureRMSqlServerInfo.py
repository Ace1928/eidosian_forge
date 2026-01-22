from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMSqlServerInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str'))
        self.results = dict(changed=False)
        self.resource_group = None
        self.server_name = None
        super(AzureRMSqlServerInfo, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_sqlserver_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_sqlserver_facts' module has been renamed to 'azure_rm_sqlserver_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.resource_group is not None and self.server_name is not None:
            self.results['servers'] = self.get()
        elif self.resource_group is not None:
            self.results['servers'] = self.list_by_resource_group()
        return self.results

    def get(self):
        """
        Gets facts of the specified SQL Server.

        :return: deserialized SQL Serverinstance state dictionary
        """
        response = None
        results = {}
        try:
            response = self.sql_client.servers.get(resource_group_name=self.resource_group, server_name=self.server_name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError:
            self.log('Could not get facts for Servers.')
        if response is not None:
            results[response.name] = self.format_results(response.as_dict())
        return results

    def list_by_resource_group(self):
        """
        Gets facts of the specified SQL Server.

        :return: deserialized SQL Serverinstance state dictionary
        """
        response = None
        results = {}
        try:
            response = self.sql_client.servers.list_by_resource_group(resource_group_name=self.resource_group)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError:
            self.log('Could not get facts for Servers.')
        if response is not None:
            for item in response:
                results[item.name] = self.format_results(item.as_dict())
        return results

    def format_results(self, response):
        administrators = response.get('administrators')
        return {'id': response.get('id'), 'name': response.get('name'), 'type': response.get('type'), 'location': response.get('location'), 'kind': response.get('kind'), 'version': response.get('version'), 'state': response.get('state'), 'tags': response.get('tags', {}), 'fully_qualified_domain_name': response.get('fully_qualified_domain_name'), 'minimal_tls_version': response.get('minimal_tls_version'), 'public_network_access': response.get('public_network_access'), 'restrict_outbound_network_access': response.get('restrict_outbound_network_access'), 'admin_username': response.get('administrator_login'), 'administrators': None if not administrators else {'administrator_type': administrators.get('administrator_type'), 'azure_ad_only_authentication': administrators.get('azure_ad_only_authentication'), 'login': administrators.get('login'), 'principal_type': administrators.get('principal_type'), 'sid': administrators.get('sid'), 'tenant_id': administrators.get('tenant_id')}}