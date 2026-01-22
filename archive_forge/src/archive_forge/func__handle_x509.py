from __future__ import absolute_import, division, print_function
from functools import partial
from ansible.errors import AnsibleFilterError
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddr_utils import _need_netaddr
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
def _handle_x509(value):
    """Convert ipv6 address into x509 format"""
    ip = netaddr.IPAddress(value)
    ipv6_oct = []
    ipv6address = ip.bits().split(':')
    for i in ipv6address:
        x = hex(int(i, 2))
        ipv6_oct.append(x.replace('0x', ''))
    return str(':'.join(ipv6_oct))