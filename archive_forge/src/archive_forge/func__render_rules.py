from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
def _render_rules(self, name, want, have, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for rules attributes.
        :param name: interface id/name.
        :param want: desired config.
        :param have: target config.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    h_rules = []
    key = 'direction'
    w_rules = want.get('rules') or []
    if have:
        h_rules = have.get('rules') or []
    for w in w_rules:
        h = search_obj_in_list(w[key], h_rules, key=key)
        if key in w:
            if opr:
                if 'name' in w and (not (h and h[key] == w[key] and (h['name'] == w['name']))):
                    commands.append(self._compute_command(afi=want['afi'], name=name, attrib=w[key], value=w['name']))
                elif not (h and key in h):
                    commands.append(self._compute_command(afi=want['afi'], name=name, attrib=w[key]))
            elif not opr:
                if not h or key not in h or ('name' in w and h and ('name' not in h)):
                    commands.append(self._compute_command(afi=want['afi'], name=name, attrib=w[key], opr=opr))
    return commands