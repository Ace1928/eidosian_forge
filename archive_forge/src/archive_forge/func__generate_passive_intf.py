from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.facts import Facts
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.ospf_interfaces import (
def _generate_passive_intf(self, data):
    cmd = 'default '
    if data['afi'] == 'ipv4':
        cmd += 'ip ospf passive-interface'
    else:
        cmd += 'ospfv3 passive-interface'
    return cmd