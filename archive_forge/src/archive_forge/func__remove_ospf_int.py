from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ospf_interfaces import (
def _remove_ospf_int(self, entry):
    int_name = entry.get('name', {})
    int_addr = entry.get('address_family', {})
    for k, addr in iteritems(int_addr):
        rem_entry = {'name': int_name, 'address_family': {'afi': k}}
        self.addcmd(rem_entry, 'ip_ospf', True)