from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_civic_address(self, name, want, have):
    commands = []
    for item in want:
        ca_type = item['ca_type']
        ca_value = item['ca_value']
        obj_in_have = search_dict_tv_in_list(ca_type, ca_value, have, 'ca_type', 'ca_value')
        if not obj_in_have:
            commands.append(self._compute_command(key=name + ' location civic-based ca-type', attrib=str(ca_type) + ' ca-value', value=ca_value))
    return commands