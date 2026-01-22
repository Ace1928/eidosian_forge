from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_global import (
def _compare_confederation_peers(self, want, have):
    """Custom handling of confederation.peers option
        :params want: the want BGP dictionary
        :params have: the have BGP dictionary
        """
    w_cpeers = want.get('bgp', {}).get('confederation', {}).get('peers', [])
    h_cpeers = have.get('bgp', {}).get('confederation', {}).get('peers', [])
    if set(w_cpeers) != set(h_cpeers):
        if self.state in ['replaced', 'deleted']:
            if h_cpeers:
                self.addcmd(have, 'bgp_confederation_peers', True)
        if w_cpeers:
            self.addcmd(want, 'bgp_confederation_peers', False)