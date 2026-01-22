from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _ipv4_hostmask(ip):
    """Test if an address is a hostmask"""
    params = {'ip': ip}
    _validate_args('ipv4_hostmask', DOCUMENTATION, params)
    try:
        ipaddr = ip_network('10.0.0.0/{ip}'.format(ip=ip))
        return str(ipaddr.hostmask) == ip
    except Exception:
        return False