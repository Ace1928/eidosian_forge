from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase, OneViewModuleResourceNotFound
def _default_bandwidth_reset(self, resource):
    if not resource:
        raise OneViewModuleResourceNotFound(self.MSG_ETHERNET_NETWORK_NOT_FOUND)
    default_connection_template = self.oneview_client.connection_templates.get_default()
    changed, connection_template = self._update_connection_template(resource, default_connection_template['bandwidth'])
    return (changed, self.MSG_CONNECTION_TEMPLATE_RESET, dict(ethernet_network_connection_template=connection_template))