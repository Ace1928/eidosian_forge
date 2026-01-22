from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_maps_match_metric(config_data):
    config_data = config_data['entries']['match']['metric']
    command = 'match metric'
    if config_data.get('value'):
        command += ' ' + config_data['value']
    return command