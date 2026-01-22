from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
def _get_ethernet_network_by_name(self, name):
    result = self.oneview_client.ethernet_networks.get_by('name', name)
    return result[0] if result else None