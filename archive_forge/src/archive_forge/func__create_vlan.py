from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def _create_vlan(self, network_domain):
    vlan = self.driver.ex_create_vlan(network_domain, self.name, self.private_ipv4_base_address, self.description, self.private_ipv4_prefix_size)
    if self.wait:
        vlan = self._wait_for_vlan_state(vlan.id, 'NORMAL')
    return vlan