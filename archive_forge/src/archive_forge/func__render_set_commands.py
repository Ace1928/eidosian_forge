from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_set_commands(self, want):
    """
        This function returns the list of commands to add attributes which are
        present in want
        :param want:
        :return: list of commands.
        """
    commands = []
    have = {}
    for key, value in iteritems(want):
        if value:
            if key == 'dest':
                commands.append(self._compute_command(dest=want['dest']))
            elif key == 'blackhole_config':
                commands.extend(self._add_blackhole(key, want, have))
            elif key == 'next_hops':
                commands.extend(self._add_next_hop(want, have))
    return commands