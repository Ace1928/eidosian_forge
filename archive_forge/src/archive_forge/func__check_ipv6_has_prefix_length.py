from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def _check_ipv6_has_prefix_length(ip_address, netmask, module):
    ip_address = _get_ipv4orv6_address(ip_address, module)
    if not isinstance(ip_address, ipaddress.IPv6Address) or isinstance(netmask, int):
        return
    if ':' in netmask:
        module.fail_json(msg='Error: only prefix_len is supported for IPv6 addresses, got %s' % netmask)