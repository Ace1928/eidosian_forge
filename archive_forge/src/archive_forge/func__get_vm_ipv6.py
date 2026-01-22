from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.common.text.converters import to_native
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from collections import namedtuple
import os
def _get_vm_ipv6(self, vm):
    nic = vm.TEMPLATE.get('NIC')
    if isinstance(nic, dict):
        nic = [nic]
    for net in nic:
        if net.get('IP6_GLOBAL'):
            return net['IP6_GLOBAL']
    return False