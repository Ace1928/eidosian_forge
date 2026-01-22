from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_ports_addrs(self, attr, w, h, opr, cmd, name, type):
    """
        This function forms the commands for port/address/network group members
        based on the 'opr'.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param cmd: commands to be prepend.
        :param name: name of group.
        :param type: group type.
        :return: generated list of commands.
        """
    commands = []
    have = []
    if w:
        want = w.get(attr) or []
    if h:
        have = h.get(attr) or []
    if want:
        if opr:
            members = list_diff_want_only(want, have)
            for member in members:
                commands.append(cmd + ' ' + name + ' ' + self._grp_type(type) + ' ' + member[self._get_mem_type(type)])
        elif not opr and have:
            members = list_diff_want_only(want, have)
            for member in members:
                commands.append(cmd + ' ' + name + ' ' + self._grp_type(type) + ' ' + member[self._get_mem_type(type)])
    return commands