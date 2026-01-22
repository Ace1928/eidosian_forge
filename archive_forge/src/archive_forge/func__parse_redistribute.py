from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _parse_redistribute(r_list):
    rcmd_list = []
    for r_dict in r_list:
        r_cmd = 'redistribute '
        r_cmd = r_cmd + r_dict['routes']
        if r_dict.get('isis_level'):
            k = re.sub('_', '-', r_dict['isis_level'])
            r_cmd = r_cmd + ' ' + k
        if r_dict.get('route_map'):
            r_cmd = r_cmd + ' route-map ' + r_dict['route_map']
        rcmd_list.append(r_cmd)
    return rcmd_list