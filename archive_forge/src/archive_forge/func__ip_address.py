from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _ip_address(ip):
    """Test if something in an IP address"""
    params = {'ip': ip}
    _validate_args('ip_address', DOCUMENTATION, params)
    try:
        ip_address(ip)
        return True
    except Exception:
        return False