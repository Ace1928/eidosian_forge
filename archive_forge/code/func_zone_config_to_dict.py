from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
def zone_config_to_dict(self, zone_config):
    return dict(name=zone_config.get('name'), private_dns_zone_id=zone_config.get('private_dns_zone_id'), record_sets=[self.record_set_to_dict(record_set) for record_set in zone_config.get('record_sets', [])])