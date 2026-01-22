from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_maximum_paths_ibgp(config_data):
    ibgp_conf = config_data.get('maximum_paths', {}).get('ibgp', {})
    if ibgp_conf:
        command = 'maximum-paths ibgp'
        if 'max_path_value' in ibgp_conf:
            command += ' ' + str(ibgp_conf['max_path_value'])
        if 'order_igp_metric' in ibgp_conf:
            command += ' order igp-metric'
        elif 'selective_order_igp_metric' in ibgp_conf:
            command += ' selective order igp-metric'
        elif 'set' in ibgp_conf.get('unequal_cost', {}):
            command += ' unequal-cost'
            if 'order_igp_metric' in ibgp_conf.get('unequal_cost', {}):
                command += ' order igp-metric'
            elif 'selective_order_igp_metric' in ibgp_conf.get('unequal_cost', {}):
                command += ' selective order igp-metric'
        return command