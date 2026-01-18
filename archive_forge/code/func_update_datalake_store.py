from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_datalake_store(self):
    self.log('Updating datalake store {0}'.format(self.name))
    parameters = self.datalake_store_models.UpdateDataLakeStoreAccountParameters()
    if self.tags:
        update_tags, self.account_dict['tags'] = self.update_tags(self.account_dict['tags'])
        if update_tags:
            self.results['changed'] = True
            parameters.tags = self.account_dict['tags']
    if self.new_tier and self.account_dict.get('new_tier') != self.new_tier:
        self.results['changed'] = True
        parameters.new_tier = self.new_tier
    if self.default_group and self.account_dict.get('default_group') != self.default_group:
        self.results['changed'] = True
        parameters.default_group = self.default_group
    if self.encryption_state and self.account_dict.get('encryption_state') != self.encryption_state:
        self.fail('Encryption type cannot be updated.')
    if self.encryption_config:
        if self.encryption_config.get('type') == 'UserManaged' and self.encryption_config.get('key_vault_meta_info') != self.account_dict.get('encryption_config').get('key_vault_meta_info'):
            self.results['changed'] = True
            key_vault_meta_info_model = self.datalake_store_models.UpdateKeyVaultMetaInfo(encryption_key_version=self.encryption_config.get('key_vault_meta_info').get('encryption_key_version'))
            encryption_config_model = self.datalake_store_models.UpdateEncryptionConfig = key_vault_meta_info_model
            parameters.encryption_config = encryption_config_model
    if self.firewall_state and self.account_dict.get('firewall_state') != self.firewall_state:
        self.results['changed'] = True
        parameters.firewall_state = self.firewall_state
    if self.firewall_allow_azure_ips and self.account_dict.get('firewall_allow_azure_ips') != self.firewall_allow_azure_ips:
        self.results['changed'] = True
        parameters.firewall_allow_azure_ips = self.firewall_allow_azure_ips
    if self.firewall_rules is not None:
        if not self.compare_lists(self.firewall_rules, self.account_dict.get('firewall_rules')):
            self.firewall_rules_model = list()
            for rule in self.firewall_rules:
                rule_model = self.datalake_store_models.UpdateFirewallRuleWithAccountParameters(name=rule.get('name'), start_ip_address=rule.get('start_ip_address'), end_ip_address=rule.get('end_ip_address'))
                self.firewall_rules_model.append(rule_model)
            self.results['changed'] = True
            parameters.firewall_rules = self.firewall_rules_model
    if self.virtual_network_rules is not None:
        if not self.compare_lists(self.virtual_network_rules, self.account_dict.get('virtual_network_rules')):
            self.virtual_network_rules_model = list()
            for vnet_rule in self.virtual_network_rules:
                vnet_rule_model = self.datalake_store_models.UpdateVirtualNetworkRuleWithAccountParameters(name=vnet_rule.get('name'), subnet_id=vnet_rule.get('subnet_id'))
                self.virtual_network_rules_model.append(vnet_rule_model)
            self.results['changed'] = True
            parameters.virtual_network_rules = self.virtual_network_rules_model
    if self.identity_model is not None:
        self.results['changed'] = True
        parameters.identity = self.identity_model
    self.log(str(parameters))
    if self.results['changed']:
        try:
            poller = self.datalake_store_client.accounts.begin_update(self.resource_group, self.name, parameters)
            self.get_poller_result(poller)
        except Exception as e:
            self.log('Error creating datalake store.')
            self.fail('Failed to create datalake store: {0}'.format(str(e)))
    return self.get_datalake_store()