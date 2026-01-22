from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
def _get_vrf_list(want):
    vrf_list = []
    if not want:
        return vrf_list
    for w in want['processes']:
        if w.get('vrf'):
            vrf_list.append(w['vrf'])
    return vrf_list