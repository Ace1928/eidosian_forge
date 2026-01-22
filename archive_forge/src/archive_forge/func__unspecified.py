from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
@_need_ipaddress
def _unspecified(ip):
    """Test for an unspecified IP address"""
    params = {'ip': ip}
    _validate_args('unspecified', DOCUMENTATION, params)
    try:
        return ip_address(ip).is_unspecified
    except Exception:
        return False