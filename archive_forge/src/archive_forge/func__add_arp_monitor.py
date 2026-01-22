from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_arp_monitor(self, updates, key, want, have):
    commands = []
    arp_monitor = updates.get(key) or {}
    diff_targets = self._get_arp_monitor_target_diff(want, have, key, 'target')
    if 'interval' in arp_monitor:
        commands.append(self._compute_command(key=want['name'] + ' arp-monitor', attrib='interval', value=str(arp_monitor['interval'])))
    if diff_targets:
        for target in diff_targets:
            commands.append(self._compute_command(key=want['name'] + ' arp-monitor', attrib='target', value=target))
    return commands