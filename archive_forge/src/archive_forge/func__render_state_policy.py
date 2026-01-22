from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_state_policy(self, attr, w, h, opr):
    """
        This function forms the commands for 'state-policy' attributes
        based on the 'opr'.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param opr: True/False.
        :return: generated list of commands.
        """
    commands = []
    have = []
    l_set = ('log', 'action', 'connection_type')
    if not opr and self._is_root_del(h, w, attr):
        commands.append(self._form_attr_cmd(attr=attr, opr=opr))
    else:
        w_sp = deepcopy(remove_empties(w))
        want = w_sp.get(attr) or []
        if h:
            have = h.get(attr) or []
        if want:
            for w in want:
                h = self.search_attrib_in_have(have, w, 'connection_type')
                for key, val in iteritems(w):
                    if val and key != 'connection_type':
                        if opr and key in l_set and (not (h and self._is_w_same(w, h, key))):
                            commands.append(self._form_attr_cmd(key=attr + ' ' + w['connection_type'], attr=key, val=self._bool_to_str(val), opr=opr))
                        elif not opr and key in l_set:
                            if not (h and self._in_target(h, key)) and (not self._is_del(l_set, h)):
                                if key == 'action':
                                    commands.append(self._form_attr_cmd(attr=attr + ' ' + w['connection_type'], opr=opr))
                                else:
                                    commands.append(self._form_attr_cmd(attr=attr + ' ' + w['connection_type'], val=self._bool_to_str(val), opr=opr))
    return commands