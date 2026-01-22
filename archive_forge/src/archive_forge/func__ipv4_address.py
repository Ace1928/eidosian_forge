from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _ipv4_address(ip):
    """Test if something in an IPv4 address"""
    params = {'ip': ip}
    _validate_args('ipv4_address', DOCUMENTATION, params)
    try:
        return ip_address(ip).version == 4
    except Exception:
        return False