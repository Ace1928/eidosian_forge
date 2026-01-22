from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_bgp_network(config_data):
    command = 'network {address}'.format(**config_data)
    if config_data.get('route_map'):
        command += ' route-map {route_map}'.format(**config_data)
    return command