from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_ping(self, attr, w, h, opr):
    """
        This function forms the commands for 'ping' attributes based on the 'opr'.
        :param attr: attribute name.
        :param w: the desired configuration.
        :param h: the target config.
        :param opr: True/False.
        :return: generated list of commands.
        """
    commands = []
    h_ping = {}
    l_set = ('all', 'broadcast')
    if h:
        h_ping = h.get(attr) or {}
    if self._is_root_del(w[attr], h_ping, attr):
        for item, value in iteritems(h[attr]):
            if not opr and item in l_set:
                commands.append(self._form_attr_cmd(attr=item, opr=opr))
    elif w[attr]:
        if h and attr in h.keys():
            h_ping = h.get(attr) or {}
        for item, value in iteritems(w[attr]):
            if opr and item in l_set and (not (h_ping and self._is_w_same(w[attr], h_ping, item))):
                commands.append(self._form_attr_cmd(attr=item, val=self._bool_to_str(value), opr=opr))
            elif not opr and item in l_set and (not (h_ping and self._is_w_same(w[attr], h_ping, item))):
                commands.append(self._form_attr_cmd(attr=item, opr=opr))
    return commands