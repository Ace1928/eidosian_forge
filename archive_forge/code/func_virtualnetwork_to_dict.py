from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def virtualnetwork_to_dict(self, vnet):
    results = dict(id=vnet.id, name=vnet.name, location=vnet.location, tags=vnet.tags, provisioning_state=vnet.provisioning_state, flow_timeout_in_minutes=vnet.flow_timeout_in_minutes)
    if vnet.dhcp_options and len(vnet.dhcp_options.dns_servers) > 0:
        results['dns_servers'] = []
        for server in vnet.dhcp_options.dns_servers:
            results['dns_servers'].append(server)
    if vnet.address_space and len(vnet.address_space.address_prefixes) > 0:
        results['address_prefixes'] = []
        for space in vnet.address_space.address_prefixes:
            results['address_prefixes'].append(space)
    if vnet.subnets and len(vnet.subnets) > 0:
        results['subnets'] = [self.subnet_to_dict(x) for x in vnet.subnets]
    return results