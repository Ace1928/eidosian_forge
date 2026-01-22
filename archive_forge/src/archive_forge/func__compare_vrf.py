from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_neighbor_address_family import (
def _compare_vrf(self, want, have):
    """Custom handling of VRFs option
        :params want: the want BGP dictionary
        :params have: the have BGP dictionary
        """
    wvrfs = want.get('vrfs', {})
    hvrfs = have.get('vrfs', {})
    for name, entry in iteritems(wvrfs):
        begin = len(self.commands)
        vrf_have = hvrfs.pop(name, {})
        self._compare_neighbors(want=entry, have=vrf_have)
        if len(self.commands) != begin:
            self.commands.insert(begin, 'vrf {0}'.format(name))
    for name, entry in iteritems(hvrfs):
        begin = len(self.commands)
        self._compare_neighbors(want={}, have=entry)
        if len(self.commands) != begin:
            self.commands.insert(begin, 'vrf {0}'.format(name))