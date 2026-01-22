from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils._text import to_native
class AzureRMStorageAccountInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str', aliases=['resource_group_name']), tags=dict(type='list', elements='str'), show_connection_string=dict(type='bool'), show_blob_cors=dict(type='bool'), show_georeplication_stats=dict(type='bool'))
        self.results = dict(changed=False, storageaccounts=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        self.show_connection_string = None
        self.show_blob_cors = None
        self.show_georeplication_stats = None
        super(AzureRMStorageAccountInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_storageaccount_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_storageaccount_facts' module has been renamed to 'azure_rm_storageaccount_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        if self.name and (not self.resource_group):
            self.fail('Parameter error: resource group required when filtering by name.')
        results = []
        if self.name:
            results = self.get_account()
        elif self.resource_group:
            results = self.list_resource_group()
        else:
            results = self.list_all()
        filtered = self.filter_tag(results)
        if is_old_facts:
            self.results['ansible_facts'] = {'azure_storageaccounts': self.serialize(filtered), 'storageaccounts': self.format_to_dict(filtered)}
        self.results['storageaccounts'] = self.format_to_dict(filtered)
        return self.results

    def get_account(self):
        self.log('Get properties for account {0}'.format(self.name))
        account = None
        try:
            expand = None
            if self.show_georeplication_stats:
                expand = 'georeplicationstats'
            account = self.storage_client.storage_accounts.get_properties(self.resource_group, self.name, expand=expand)
            return [account]
        except Exception as exc:
            if 'InvalidAccountType' in str(exc) or 'LastSyncTimeUnavailable' in str(exc):
                account = self.storage_client.storage_accounts.get_properties(self.resource_group, self.name)
                return [account]
            if 'AuthorizationFailed' in str(exc):
                self.fail('Error authenticating with the Azure storage API. {0}'.format(str(exc)))
        return []

    def list_resource_group(self):
        self.log('List items')
        try:
            response = self.storage_client.storage_accounts.list_by_resource_group(self.resource_group)
        except Exception as exc:
            self.fail('Error listing for resource group {0} - {1}'.format(self.resource_group, str(exc)))
        return response

    def list_all(self):
        self.log('List all items')
        try:
            response = self.storage_client.storage_accounts.list()
        except Exception as exc:
            self.fail('Error listing all items - {0}'.format(str(exc)))
        return response

    def filter_tag(self, raw):
        return [item for item in raw if self.has_tags(item.tags, self.tags)]

    def serialize(self, raw):
        return [self.serialize_obj(item, AZURE_OBJECT_CLASS) for item in raw]

    def format_to_dict(self, raw):
        return [self.account_obj_to_dict(item) for item in raw]

    def account_obj_to_dict(self, account_obj):
        account_dict = dict(id=account_obj.id, name=account_obj.name, location=account_obj.location, failover_in_progress=account_obj.failover_in_progress if account_obj.failover_in_progress is not None else False, access_tier=account_obj.access_tier if account_obj.access_tier is not None else None, account_type=account_obj.sku.name, kind=account_obj.kind if account_obj.kind else None, provisioning_state=account_obj.provisioning_state, secondary_location=account_obj.secondary_location, status_of_primary=account_obj.status_of_primary if account_obj.status_of_primary is not None else None, status_of_secondary=account_obj.status_of_secondary if account_obj.status_of_secondary is not None else None, primary_location=account_obj.primary_location, https_only=account_obj.enable_https_traffic_only, minimum_tls_version=account_obj.minimum_tls_version, public_network_access=account_obj.public_network_access, allow_blob_public_access=account_obj.allow_blob_public_access, is_hns_enabled=account_obj.is_hns_enabled if account_obj.is_hns_enabled else False, static_website=dict(enabled=False, index_document=None, error_document404_path=None))
        account_dict['geo_replication_stats'] = None
        if account_obj.geo_replication_stats is not None:
            account_dict['geo_replication_stats'] = dict(status=account_obj.geo_replication_stats.status, can_failover=account_obj.geo_replication_stats.can_failover, last_sync_time=account_obj.geo_replication_stats.last_sync_time)
        id_dict = self.parse_resource_to_dict(account_obj.id)
        account_dict['resource_group'] = id_dict.get('resource_group')
        account_key = self.get_connectionstring(account_dict['resource_group'], account_dict['name'])
        account_dict['custom_domain'] = None
        if account_obj.custom_domain:
            account_dict['custom_domain'] = dict(name=account_obj.custom_domain.name, use_sub_domain=account_obj.custom_domain.use_sub_domain)
        account_dict['network_acls'] = None
        if account_obj.network_rule_set:
            account_dict['network_acls'] = dict(bypass=account_obj.network_rule_set.bypass, default_action=account_obj.network_rule_set.default_action, ip_rules=account_obj.network_rule_set.ip_rules)
            if account_obj.network_rule_set.virtual_network_rules:
                account_dict['network_acls']['virtual_network_rules'] = []
                for rule in account_obj.network_rule_set.virtual_network_rules:
                    account_dict['network_acls']['virtual_network_rules'].append(dict(id=rule.virtual_network_resource_id, action=rule.action))
            if account_obj.network_rule_set.ip_rules:
                account_dict['network_acls']['ip_rules'] = []
                for rule in account_obj.network_rule_set.ip_rules:
                    account_dict['network_acls']['ip_rules'].append(dict(value=rule.ip_address_or_range, action=rule.action))
        account_dict['primary_endpoints'] = None
        if account_obj.primary_endpoints:
            account_dict['primary_endpoints'] = dict(blob=self.format_endpoint_dict(account_dict['name'], account_key[0], account_obj.primary_endpoints.blob, 'blob'), file=self.format_endpoint_dict(account_dict['name'], account_key[0], account_obj.primary_endpoints.file, 'file'), queue=self.format_endpoint_dict(account_dict['name'], account_key[0], account_obj.primary_endpoints.queue, 'queue'), table=self.format_endpoint_dict(account_dict['name'], account_key[0], account_obj.primary_endpoints.table, 'table'))
            if account_key[0]:
                account_dict['primary_endpoints']['key'] = '{0}'.format(account_key[0])
        account_dict['secondary_endpoints'] = None
        if account_obj.secondary_endpoints:
            account_dict['secondary_endpoints'] = dict(blob=self.format_endpoint_dict(account_dict['name'], account_key[1], account_obj.primary_endpoints.blob, 'blob'), file=self.format_endpoint_dict(account_dict['name'], account_key[1], account_obj.primary_endpoints.file, 'file'), queue=self.format_endpoint_dict(account_dict['name'], account_key[1], account_obj.primary_endpoints.queue, 'queue'), table=self.format_endpoint_dict(account_dict['name'], account_key[1], account_obj.primary_endpoints.table, 'table'))
            if account_key[1]:
                account_dict['secondary_endpoints']['key'] = '{0}'.format(account_key[1])
        account_dict['tags'] = None
        if account_obj.tags:
            account_dict['tags'] = account_obj.tags
        blob_mgmt_props = self.get_blob_mgmt_props(account_dict['resource_group'], account_dict['name'])
        if blob_mgmt_props and blob_mgmt_props.cors and blob_mgmt_props.cors.cors_rules:
            account_dict['blob_cors'] = [dict(allowed_origins=to_native(x.allowed_origins), allowed_methods=to_native(x.allowed_methods), max_age_in_seconds=x.max_age_in_seconds, exposed_headers=to_native(x.exposed_headers), allowed_headers=to_native(x.allowed_headers)) for x in blob_mgmt_props.cors.cors_rules]
        blob_client_props = self.get_blob_client_props(account_dict['resource_group'], account_dict['name'], account_dict['kind'])
        if blob_client_props and blob_client_props['static_website']:
            static_website = blob_client_props['static_website']
            account_dict['static_website'] = dict(enabled=static_website.enabled, index_document=static_website.index_document, error_document404_path=static_website.error_document404_path)
        account_dict['encryption'] = dict()
        if account_obj.encryption:
            account_dict['encryption']['require_infrastructure_encryption'] = account_obj.encryption.require_infrastructure_encryption
            account_dict['encryption']['key_source'] = account_obj.encryption.key_source
            if account_obj.encryption.services:
                account_dict['encryption']['services'] = dict()
                if account_obj.encryption.services.file:
                    account_dict['encryption']['services']['file'] = dict(enabled=True)
                if account_obj.encryption.services.table:
                    account_dict['encryption']['services']['table'] = dict(enabled=True)
                if account_obj.encryption.services.queue:
                    account_dict['encryption']['services']['queue'] = dict(enabled=True)
                if account_obj.encryption.services.blob:
                    account_dict['encryption']['services']['blob'] = dict(enabled=True)
        return account_dict

    def format_endpoint_dict(self, name, key, endpoint, storagetype, protocol='https'):
        result = dict(endpoint=endpoint)
        if key:
            result['connectionstring'] = 'DefaultEndpointsProtocol={0};EndpointSuffix={1};AccountName={2};AccountKey={3};{4}Endpoint={5}'.format(protocol, self._cloud_environment.suffixes.storage_endpoint, name, key, str.title(storagetype), endpoint)
        return result

    def get_blob_mgmt_props(self, resource_group, name):
        if not self.show_blob_cors:
            return None
        try:
            return self.storage_client.blob_services.get_service_properties(resource_group, name)
        except Exception:
            pass
        return None

    def get_blob_client_props(self, resource_group, name, kind):
        if kind == 'FileStorage':
            return None
        try:
            return self.get_blob_service_client(resource_group, name).get_service_properties()
        except Exception:
            pass
        return None

    def get_connectionstring(self, resource_group, name):
        keys = ['', '']
        if not self.show_connection_string:
            return keys
        try:
            cred = self.storage_client.storage_accounts.list_keys(resource_group, name)
            try:
                keys = [cred.keys[0].value, cred.keys[1].value]
            except AttributeError:
                keys = [cred.key1, cred.key2]
        except Exception:
            pass
        return keys