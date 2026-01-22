from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.bgp_templates import (
def _afs_compare(self, want, have):
    for name, wentry in iteritems(want):
        begin = len(self.commands)
        self._af_compare(want=wentry, have=have.pop(name, {}))
        if begin != len(self.commands):
            self.commands.insert(begin, self._tmplt.render(wentry, 'address_family', False))
    for name, hentry in iteritems(have):
        self.commands.append(self._tmplt.render(hentry, 'address_family', True))