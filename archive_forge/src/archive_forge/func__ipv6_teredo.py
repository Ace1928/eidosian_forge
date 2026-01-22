from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _ipv6_teredo(ip):
    """Test if something is an IPv6 teredo address"""
    params = {'ip': ip}
    _validate_args('ipv6_teredo', DOCUMENTATION, params)
    try:
        if ip_address(ip).teredo is None:
            return False
        return True
    except Exception:
        return False