from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.l3_acls.l3_acls import L3_aclsArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
@staticmethod
def _convert_ip_addr_to_spec_fmt(ip_addr, is_ipv4=False):
    spec_fmt = {}
    if ip_addr is not None:
        ip_addr = ip_addr.lower()
        if is_ipv4:
            host_mask = IPV4_HOST_MASK
        else:
            host_mask = IPV6_HOST_MASK
        if ip_addr.endswith(host_mask):
            spec_fmt['host'] = ip_addr.replace(host_mask, '')
        else:
            spec_fmt['prefix'] = ip_addr
    else:
        spec_fmt['any'] = True
    return spec_fmt