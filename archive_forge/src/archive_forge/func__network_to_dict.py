from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import HAS_LIBCLOUD, DimensionDataModule
from ansible.module_utils.common.text.converters import to_native
def _network_to_dict(self, network):
    network_dict = dict(id=network.id, name=network.name, description=network.description)
    if isinstance(network.location, NodeLocation):
        network_dict['location'] = network.location.id
    else:
        network_dict['location'] = network.location
    if self.mcp_version == '1.0':
        network_dict['private_net'] = network.private_net
        network_dict['multicast'] = network.multicast
        network_dict['status'] = None
    else:
        network_dict['private_net'] = None
        network_dict['multicast'] = None
        network_dict['status'] = network.status
    return network_dict