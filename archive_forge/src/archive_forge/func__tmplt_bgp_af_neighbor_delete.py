from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_af_neighbor_delete(config_data):
    afi = config_data['neighbors']['address_family']['afi'] + '-unicast'
    command = 'protocols bgp {as_number} '.format(**config_data)
    command += 'neighbor {neighbor_address} address-family '.format(**config_data['neighbors']) + afi
    config_data = config_data['neighbors']['address_family']
    if config_data.get('allowas_in'):
        command += ' allowas-in'
    elif config_data.get('as_override'):
        command += ' as-override'
    elif config_data.get('attribute_unchanged'):
        command += ' attribute-unchanged'
    elif config_data.get('capability'):
        command += ' capability'
    elif config_data.get('default_originate'):
        command += ' default-originate'
    elif config_data.get('maximum_prefix'):
        command += ' maximum-prefix'
    elif config_data.get('nexthop_local'):
        command += ' nexthop-local'
    elif config_data.get('nexthop_self'):
        command += ' nexthop-self'
    elif config_data.get('peer_group'):
        command += ' peer-group'
    elif config_data.get('remote_private_as'):
        command += ' remote-private-as'
    elif config_data.get('route_reflector_client'):
        command += ' route-reflector-client'
    elif config_data.get('route_server_client'):
        command += ' route-server-client'
    elif config_data.get('soft_reconfiguration'):
        command += ' soft-reconfiguration'
    elif config_data.get('unsuppress_map'):
        command += ' unsuppress-map'
    elif config_data.get('weight'):
        command += ' weight'
    elif config_data.get('filter_list'):
        command += ' filter-list'
    elif config_data.get('prefix_list'):
        command += ' prefix-list'
    elif config_data.get('distribute_list'):
        command += ' distribute-list'
    elif config_data.get('route_map'):
        command += ' route-map'
    return command