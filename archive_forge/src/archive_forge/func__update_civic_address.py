from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _update_civic_address(self, name, want, have):
    commands = []
    for item in have:
        ca_type = item['ca_type']
        ca_value = item['ca_value']
        in_want = search_dict_tv_in_list(ca_type, ca_value, want, 'ca_type', 'ca_value')
        if not in_want:
            commands.append(self._compute_command(name, 'location civic-based ca-type', str(ca_type), remove=True))
    return commands