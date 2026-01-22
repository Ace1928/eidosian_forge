from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_confederation_peers(config_data):
    cmds = []
    base_cmd = 'bgp confederation peers '
    peers = config_data.get('bgp', {}).get('confederation', {}).get('peers')
    if peers:
        for peer in peers:
            cmds.append(base_cmd + str(peer))
    return cmds