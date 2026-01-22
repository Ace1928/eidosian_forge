from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.bgp_address_family import (
def _remove_af(self, want_af, have_af, vrf=None, remove=False, purge=False):
    cur_ptr = len(self.commands)
    for k, v in iteritems(have_af):
        if any((remove and k in want_af, not remove and k not in want_af, purge)):
            self.addcmd(v, 'address_family', True)
    if cur_ptr < len(self.commands) and vrf:
        self.commands.insert(cur_ptr, 'vrf {0}'.format(vrf))
        self.commands.append('exit')