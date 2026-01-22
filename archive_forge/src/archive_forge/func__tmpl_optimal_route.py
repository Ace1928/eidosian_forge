from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_optimal_route(config_data):
    orr_conf = config_data.get('optimal_route_reflection', {})
    if orr_conf:
        command = 'optimal-route-reflection'
        if 'group_name' in orr_conf:
            command += ' ' + str(orr_conf['value'])
        if 'primary_address' in orr_conf:
            command += ' ' + orr_conf['primary_address']
        if 'secondary_address' in orr_conf:
            command += ' ' + orr_conf['secondary_address']
        return command