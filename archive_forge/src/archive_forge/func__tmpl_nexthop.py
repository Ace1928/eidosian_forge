from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmpl_nexthop(config_data):
    nexthop_conf = config_data.get('nexthop', {})
    commands = []
    if nexthop_conf:
        if 'resolution_prefix_length_minimum' in nexthop_conf:
            command = 'nexthop resolution prefix-length minimum ' + str(nexthop_conf['resolution_prefix_length_minimum'])
            commands.append(command)
        if 'trigger_delay_critical' in nexthop_conf:
            command = 'nexthop trigger-delay critical ' + str(nexthop_conf['trigger_delay_non_critical'])
            commands.append(command)
        if 'trigger_delay_non_critical' in nexthop_conf:
            command = 'nexthop trigger-delay non-critical ' + str(nexthop_conf['trigger_delay_non_critical'])
            commands.append(command)
        if 'route_policy' in nexthop_conf:
            command += ' route-policy ' + nexthop_conf['route_policy']
    return commands