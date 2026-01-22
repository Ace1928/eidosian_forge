from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_maximum_paths_eibgp(config_data):
    eibgp_conf = config_data.get('maximum_paths', {}).get('eibgp', {})
    if eibgp_conf:
        command = 'maximum-paths ebgp'
        if 'max_path_value' in eibgp_conf:
            command += ' ' + str(eibgp_conf['max_path_value'])
        if 'order_igp_metric' in eibgp_conf:
            command += ' order igp-metric'
        elif 'selective_order_igp_metric' in eibgp_conf:
            command += ' selective order igp-metric'
        return command