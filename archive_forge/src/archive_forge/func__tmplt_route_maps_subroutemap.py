from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_maps_subroutemap(config_data):
    command = ''
    if config_data['entries'].get('sub_route_map'):
        command = 'sub-route-map ' + config_data['entries']['sub_route_map']['name']
    if config_data['entries']['sub_route_map'].get('invert_result'):
        command += ' invert-result'
    return command