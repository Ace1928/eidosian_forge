from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_match_ip_multicast(data):
    cmd = 'match ip multicast'
    multicast = data['match']['ip']['multicast']
    if 'source' in multicast:
        cmd += ' source {source}'.format(**multicast)
    if 'prefix' in multicast.get('group', {}):
        cmd += ' group {prefix}'.format(**multicast['group'])
    else:
        if 'first' in multicast.get('group_range', {}):
            cmd += ' group-range {first}'.format(**multicast['group_range'])
        if 'last' in multicast.get('group_range', {}):
            cmd += ' to {last}'.format(**multicast['group_range'])
    if 'rp' in multicast:
        cmd += ' rp {prefix}'.format(**multicast['rp'])
        if 'rp_type' in multicast['rp']:
            cmd += ' rp-type {rp_type}'.format(**multicast['rp'])
    return cmd