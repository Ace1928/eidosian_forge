from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_route_map_match_invert_aggregator_role(config_data):
    config_data = config_data['entries']['match']['invert_result']['aggregate_role']
    command = 'match invert-result as-path aggregate-role contributor'
    if config_data.get('route_map'):
        command += ' aggregator-attributes ' + config_data['route_map']
    return command