from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def validate_and_compress_ip_address(ip_address, module):
    """
    0's in IPv6 addresses can be compressed to save space
    This will be a noop for IPv4 address
    In addition, it makes sure the address is in a valid format
    """
    return str(_get_ipv4orv6_address(ip_address, module))