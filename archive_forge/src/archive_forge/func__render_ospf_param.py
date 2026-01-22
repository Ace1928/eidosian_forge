from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_ospf_param(self, want, have, opr=True):
    """
        This function forms the set/delete commands for ospf leaf attributes
        and triggers the process for other child attributes.
        for firewall_global attributes.
        :param w: the desired config.
        :param h: the target config.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    w = deepcopy(remove_empties(want))
    leaf = ('default_metric', 'log_adjacency_changes')
    if w:
        for key, val in iteritems(w):
            if opr and key in leaf and (not _is_w_same(w, have, key)):
                commands.append(self._form_attr_cmd(attr=key, val=_bool_to_str(val), opr=opr))
            elif not opr and key in leaf and (not _in_target(have, key)):
                commands.append(self._form_attr_cmd(attr=key, val=_bool_to_str(val), opr=opr))
            else:
                commands.extend(self._render_child_param(w, have, key, opr))
    return commands