from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import load_config, run_commands
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch import build_aggregate_spec, map_params_to_obj
from ansible_collections.community.network.plugins.module_utils.network.edgeswitch.edgeswitch_interface import InterfaceConfiguration, merge_interfaces
def unrange(vlans):
    res = []
    for vlan in vlans:
        match = re.match('(\\d+)-(\\d+)', vlan)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            for vlan_id in range(start, end + 1):
                res.append(str(vlan_id))
        else:
            res.append(vlan)
    return res