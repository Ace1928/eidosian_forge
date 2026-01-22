from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
@_need_ipaddress
def _loopback(ip):
    """Test if an IP address is a loopback"""
    params = {'ip': ip}
    _validate_args('loopback', DOCUMENTATION, params)
    try:
        return ip_address(ip).is_loopback
    except Exception:
        return False