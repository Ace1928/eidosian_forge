from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.utils.utils import (
def _v4_cmds(self, want, have, state=None):
    """Helper method for processing ipv4 changes.
        This is needed to handle primary/secondary address changes, which require a specific sequence when changing.
        """
    for i in want:
        i['tag'] = i.get('tag')
    for i in have:
        i['tag'] = i.get('tag')
    merged = True if state == 'merged' else False
    replaced = True if state == 'replaced' else False
    overridden = True if state == 'overridden' else False
    sec_w = [i for i in want if i.get('secondary')]
    sec_h = [i for i in have if i.get('secondary')]
    pri_w = [i for i in want if not i.get('secondary')]
    pri_h = [i for i in have if not i.get('secondary')]
    pri_w = pri_w[0] if pri_w else {}
    pri_h = pri_h[0] if pri_h else {}
    cmds = []
    if pri_h and (not pri_w) and (replaced or overridden):
        cmds.append('no ip address')
        return cmds
    sec_to_rmv = []
    sec_diff = self.diff_list_of_dicts(sec_h, sec_w)
    for i in sec_diff:
        if overridden or [w for w in sec_w if w['address'] == i['address']]:
            sec_to_rmv.append(i['address'])
    if pri_w and [h for h in sec_h if h['address'] == pri_w['address']]:
        if not overridden:
            sec_to_rmv.append(pri_w['address'])
    cmds.extend(['no ip address %s secondary' % i for i in sec_to_rmv])
    if pri_w:
        diff = dict(set(pri_w.items()) - set(pri_h.items()))
        if diff:
            addr = diff.get('address') or pri_w.get('address')
            cmd = 'ip address %s' % addr
            tag = diff.get('tag') or pri_w.get('tag')
            cmd += ' tag %s' % tag if tag else ''
            cmds.append(cmd)
    sec_w_to_chg = self.diff_list_of_dicts(sec_w, sec_h)
    for i in sec_w_to_chg:
        cmd = 'ip address %s secondary' % i['address']
        cmd += ' tag %s' % i['tag'] if i['tag'] else ''
        cmds.append(cmd)
    return cmds