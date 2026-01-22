from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_icmp_redirects(self, attr, w, h, opr):
    """
        This function forms the commands for 'icmp_redirects' attributes
        based on the 'opr'.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param opr: True/False.
        :return: generated list of commands.
        """
    commands = []
    h_red = {}
    l_set = ('send', 'receive')
    if w[attr]:
        if h and attr in h.keys():
            h_red = h.get(attr) or {}
        for item, value in iteritems(w[attr]):
            if opr and item in l_set and (not (h_red and self._is_w_same(w[attr], h_red, item))):
                commands.append(self._form_attr_cmd(attr=item, val=self._bool_to_str(value), opr=opr))
            elif not opr and item in l_set and (not (h_red and self._is_w_same(w[attr], h_red, item))):
                commands.append(self._form_attr_cmd(attr=item, opr=opr))
    return commands