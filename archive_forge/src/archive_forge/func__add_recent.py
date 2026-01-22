from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_recent(self, attr, w, h, cmd, opr):
    """
        This function forms the command for 'recent' attributes based on the 'opr'.
        :param attr: attribute name.
        :param w: base config.
        :param h: target config.
        :param cmd: commands to be prepend.
        :return: generated list of commands.
        """
    commands = []
    h_recent = {}
    l_set = ('count', 'time')
    if w[attr]:
        if h and attr in h.keys():
            h_recent = h.get(attr) or {}
        for item, val in iteritems(w[attr]):
            if opr and item in l_set and (not (h_recent and self._is_w_same(w[attr], h_recent, item))):
                commands.append(cmd + (' ' + attr + ' ' + item + ' ' + str(val)))
            elif not opr and item in l_set and (not (h_recent and self._in_target(h_recent, item))):
                commands.append(cmd + (' ' + attr + ' ' + item))
    return commands