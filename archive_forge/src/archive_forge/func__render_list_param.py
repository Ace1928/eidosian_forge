from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_list_param(self, attr, want, have, cmd=None, opr=True):
    """
        This function forms the commands for passed target list attributes'.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param cmd: commands to be prepend.
        :param opr: True/False.
        :return: generated list of commands.
        """
    commands = []
    h = []
    if want:
        w = want.get(attr) or []
    if have:
        h = have.get(attr) or []
    if not cmd:
        cmd = self._compute_command(opr=opr)
    if w:
        if opr:
            members = list_diff_want_only(w, h)
            for member in members:
                command = cmd + attr.replace('_', '-') + ' '
                if attr == 'network':
                    command += member['address']
                else:
                    command += member
                commands.append(command)
        elif not opr:
            if h:
                for member in w:
                    if attr == 'network':
                        if not self.search_obj_in_have(h, member, 'address'):
                            commands.append(cmd + attr.replace('_', '-') + ' ' + member['address'])
                    elif member not in h:
                        commands.append(cmd + attr.replace('_', '-') + ' ' + member)
            else:
                commands.append(cmd + ' ' + attr.replace('_', '-'))
    return commands