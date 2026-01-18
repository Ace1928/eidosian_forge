from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def validate_primary_ips(self, want):
    error_intf = {}
    for l3 in want:
        l3_interface_name = l3.get('name')
        ipv4_addrs = []
        if l3.get('ipv4') and l3['ipv4'].get('addresses'):
            ipv4_addrs = l3['ipv4']['addresses']
        if ipv4_addrs:
            ipv4_pri_addrs = [addr['address'] for addr in ipv4_addrs if not addr['secondary']]
            if len(ipv4_pri_addrs) > 1:
                error_intf[l3_interface_name] = ipv4_pri_addrs
    if error_intf:
        err = 'Multiple ipv4 primary ips found! ' + str(error_intf)
        self._module.fail_json(msg=str(err), code=300)