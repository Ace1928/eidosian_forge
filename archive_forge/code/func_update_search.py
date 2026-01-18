from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_search(self):
    self.log('Updating search {0}'.format(self.name))
    self.check_values(self.hosting_mode or self.account_dict.get('hosting_mode'), self.sku or self.account_dict.get('sku'), self.partition_count or self.account_dict.get('partition_count'), self.replica_count or self.account_dict.get('replica_count'))
    search_update_model = self.search_client.services.models.SearchServiceUpdate(location=self.location, hosting_mode=None, partition_count=None, public_network_access=None, replica_count=None, sku=self.search_client.services.models.Sku(name=self.account_dict.get('sku')))
    if self.hosting_mode and self.account_dict.get('hosting_mode') != self.hosting_mode:
        self.fail('Updating hosting_mode of an existing search service is not allowed.')
    if self.identity and self.account_dict.get('identity').get('type') != self.identity:
        self.results['changed'] = True
        search_update_model.identity = self.search_client.services.models.Identity(type=self.identity)
    if self.network_rule_set:
        for rule in self.network_rule_set:
            if len(self.network_rule_set) != len(self.account_dict.get('network_rule_set')) or rule not in self.account_dict.get('network_rule_set'):
                self.results['changed'] = True
            self.firewall_list.append(self.search_client.services.models.IpRule(value=rule))
            search_update_model.network_rule_set = dict(ip_rules=self.firewall_list)
    if self.partition_count and self.account_dict.get('partition_count') != self.partition_count:
        self.results['changed'] = True
        search_update_model.partition_count = self.partition_count
    if self.public_network_access and self.account_dict.get('public_network_access').lower() != self.public_network_access.lower():
        self.results['changed'] = True
        search_update_model.public_network_access = self.public_network_access
    if self.replica_count and self.account_dict.get('replica_count') != self.replica_count:
        self.results['changed'] = True
        search_update_model.replica_count = self.replica_count
    if self.sku and self.account_dict.get('sku') != self.sku:
        self.fail('Updating sku of an existing search service is not allowed.')
    if self.tags and self.account_dict.get('tags') != self.tags:
        self.results['changed'] = True
        search_update_model.tags = self.tags
    self.log('Updating search {0}'.format(self.name))
    try:
        if self.results['changed']:
            poller = self.search_client.services.begin_create_or_update(self.resource_group, self.name, search_update_model)
            self.get_poller_result(poller)
    except Exception as e:
        self.fail('Failed to update the search: {0}'.format(str(e)))
    return self.get_search()