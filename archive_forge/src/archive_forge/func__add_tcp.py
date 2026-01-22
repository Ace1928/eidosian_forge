from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_tcp(self, attr, w, h, cmd, opr):
    """
        This function forms the commands for 'tcp' attributes based on the 'opr'.
        :param attr: attribute name.
        :param w: base config.
        :param h: target config.
        :param cmd: commands to be prepend.
        :return: generated list of commands.
        """
    h_tcp = {}
    commands = []
    if w[attr]:
        key = 'flags'
        flags = w[attr].get(key) or {}
        if flags:
            if h and key in h[attr].keys():
                h_tcp = h[attr].get(key) or {}
            if flags:
                if opr and (not (h_tcp and self._is_w_same(w[attr], h[attr], key))):
                    commands.append(cmd + (' ' + attr + ' ' + key + ' ' + flags))
                if not opr and (not (h_tcp and self._is_w_same(w[attr], h[attr], key))):
                    commands.append(cmd + (' ' + attr + ' ' + key + ' ' + flags))
    return commands