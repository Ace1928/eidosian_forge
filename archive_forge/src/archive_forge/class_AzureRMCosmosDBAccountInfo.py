from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake
class AzureRMCosmosDBAccountInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str'), name=dict(type='str'), tags=dict(type='list', elements='str'), retrieve_keys=dict(type='str', choices=['all', 'readonly']), retrieve_connection_strings=dict(type='bool'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.name = None
        self.tags = None
        self.retrieve_keys = None
        self.retrieve_connection_strings = None
        super(AzureRMCosmosDBAccountInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_cosmosdbaccount_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_cosmosdbaccount_facts' module has been renamed to 'azure_rm_cosmosdbaccount_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(CosmosDBManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name is not None:
            self.results['accounts'] = self.get()
        elif self.resource_group is not None:
            self.results['accounts'] = self.list_by_resource_group()
        else:
            self.results['accounts'] = self.list_all()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.database_accounts.get(resource_group_name=self.resource_group, account_name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Database Account.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def list_by_resource_group(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.database_accounts.list_by_resource_group(resource_group_name=self.resource_group)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Database Account.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def list_all(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.database_accounts.list()
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Database Account.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'id': d.get('id'), 'resource_group': self.parse_resource_to_dict(d.get('id')).get('resource_group'), 'name': d.get('name', None), 'location': d.get('location', '').replace(' ', '').lower(), 'kind': _camel_to_snake(d.get('kind', None)), 'consistency_policy': {'default_consistency_level': _camel_to_snake(d['consistency_policy']['default_consistency_level']), 'max_interval_in_seconds': d['consistency_policy']['max_interval_in_seconds'], 'max_staleness_prefix': d['consistency_policy']['max_staleness_prefix']}, 'failover_policies': [{'name': fp['location_name'].replace(' ', '').lower(), 'failover_priority': fp['failover_priority'], 'id': fp['id']} for fp in d['failover_policies']], 'read_locations': [{'name': rl['location_name'].replace(' ', '').lower(), 'failover_priority': rl['failover_priority'], 'id': rl['id'], 'document_endpoint': rl['document_endpoint'], 'provisioning_state': rl['provisioning_state']} for rl in d['read_locations']], 'write_locations': [{'name': wl['location_name'].replace(' ', '').lower(), 'failover_priority': wl['failover_priority'], 'id': wl['id'], 'document_endpoint': wl['document_endpoint'], 'provisioning_state': wl['provisioning_state']} for wl in d['write_locations']], 'database_account_offer_type': d.get('database_account_offer_type'), 'enable_free_tier': d.get('enable_free_tier'), 'ip_rules': [ip['ip_address_or_range'] for ip in d.get('ip_rules', [])], 'ip_range_filter': ','.join([ip['ip_address_or_range'] for ip in d.get('ip_rules', [])]), 'is_virtual_network_filter_enabled': d.get('is_virtual_network_filter_enabled'), 'enable_automatic_failover': d.get('enable_automatic_failover'), 'enable_cassandra': 'EnableCassandra' in d.get('capabilities', []), 'enable_table': 'EnableTable' in d.get('capabilities', []), 'enable_gremlin': 'EnableGremlin' in d.get('capabilities', []), 'mongo_version': d.get('api_properties', {}).get('server_version'), 'public_network_access': d.get('public_network_access'), 'virtual_network_rules': d.get('virtual_network_rules'), 'enable_multiple_write_locations': d.get('enable_multiple_write_locations'), 'document_endpoint': d.get('document_endpoint'), 'provisioning_state': d.get('provisioning_state'), 'tags': d.get('tags', None)}
        if self.retrieve_keys == 'all':
            keys = self.mgmt_client.database_accounts.list_keys(resource_group_name=self.resource_group, account_name=self.name)
            d['primary_master_key'] = keys.primary_master_key
            d['secondary_master_key'] = keys.secondary_master_key
            d['primary_readonly_master_key'] = keys.primary_readonly_master_key
            d['secondary_readonly_master_key'] = keys.secondary_readonly_master_key
        elif self.retrieve_keys == 'readonly':
            keys = self.mgmt_client.database_accounts.get_read_only_keys(resource_group_name=self.resource_group, account_name=self.name)
            d['primary_readonly_master_key'] = keys.primary_readonly_master_key
            d['secondary_readonly_master_key'] = keys.secondary_readonly_master_key
        if self.retrieve_connection_strings:
            connection_strings = self.mgmt_client.database_accounts.list_connection_strings(resource_group_name=self.resource_group, account_name=self.name)
            d['connection_strings'] = connection_strings.as_dict()
        return d