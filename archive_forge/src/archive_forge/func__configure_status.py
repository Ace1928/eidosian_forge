from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _configure_status(self, name, want_item, have_item):
    commands = []
    if is_dict_element_present(have_item, 'enable'):
        temp_have_item = False
    else:
        temp_have_item = True
    if want_item['enable'] != temp_have_item:
        if want_item['enable']:
            commands.append(self._compute_command(name, value='disable', remove=True))
        else:
            commands.append(self._compute_command(name, value='disable'))
    return commands