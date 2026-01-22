from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def _update_version2_resources(self, resources):
    allocations = list()
    for resource in resources:
        resource_cfg = self.find_netioc_by_key(resource['name'])
        allocation = vim.DVSNetworkResourcePoolConfigSpec()
        allocation.allocationInfo = vim.DVSNetworkResourcePoolAllocationInfo()
        allocation.key = resource['name']
        allocation.configVersion = resource_cfg.configVersion
        if 'limit' in resource:
            allocation.allocationInfo.limit = resource['limit']
        if 'shares_level' in resource:
            allocation.allocationInfo.shares = vim.SharesInfo()
            allocation.allocationInfo.shares.level = resource['shares_level']
            if 'shares' in resource and resource['shares_level'] == 'custom':
                allocation.allocationInfo.shares.shares = resource['shares']
        allocations.append(allocation)
    self.dvs.UpdateNetworkResourcePool(allocations)