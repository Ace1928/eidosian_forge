from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospfv3 import (
def _af_compare(self, want, have):
    parsers = ['default_information.originate', 'distance', 'maximum_paths', 'table_map', 'timers.throttle.spf']
    waf = want.get('address_family', {})
    haf = have.get('address_family', {})
    cmd_ptr = len(self.commands)
    self._af_areas_compare(want=waf, have=haf)
    self._af_compare_lists(want=waf, have=haf)
    self.compare(parsers=parsers, want=waf, have=haf)
    cmd_ptr_nxt = len(self.commands)
    if cmd_ptr < cmd_ptr_nxt:
        self.commands.insert(cmd_ptr, 'address-family ipv6 unicast')