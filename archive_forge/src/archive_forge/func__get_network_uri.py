from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
def _get_network_uri(self, network_name_or_uri):
    if network_name_or_uri.startswith('/rest/ethernet-networks'):
        return network_name_or_uri
    else:
        enet_network = self._get_ethernet_network_by_name(network_name_or_uri)
        if enet_network:
            return enet_network['uri']
        else:
            raise OneViewModuleResourceNotFound(self.MSG_ETHERNET_NETWORK_NOT_FOUND + network_name_or_uri)