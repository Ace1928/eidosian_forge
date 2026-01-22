from __future__ import absolute_import, division, print_function
import socket
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _resolvable(host):
    """Test if an IP or name can be resolved via /etc/hosts or DNS"""
    params = {'host': host}
    _validate_args('resolvable', DOCUMENTATION, params)
    try:
        ipaddress.ip_address(host)
        ip = True
    except Exception:
        ip = False
    if ip:
        try:
            socket.gethostbyaddr(host)
            return True
        except Exception:
            return False
    else:
        try:
            socket.getaddrinfo(host, None)
            return True
        except Exception:
            return False