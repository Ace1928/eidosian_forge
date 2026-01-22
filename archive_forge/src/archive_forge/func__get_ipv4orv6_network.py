from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def _get_ipv4orv6_network(ip_address, netmask, strict, module):
    """
    return IPV4Network or IPV6Network object
    """
    _check_ipaddress_is_present(module)
    ip_addr = u'%s/%s' % (ip_address, netmask) if netmask is not None else u'%s' % ip_address
    try:
        return ipaddress.ip_network(ip_addr, strict)
    except ValueError as exc:
        error = 'Error: Invalid IP network value %s' % ip_addr
        if 'has host bits set' in to_native(exc):
            error += '.  Please specify a network address without host bits set'
        elif netmask is not None:
            error += '.  Check address and netmask values'
        error += ': %s.' % to_native(exc)
        module.fail_json(msg=error)