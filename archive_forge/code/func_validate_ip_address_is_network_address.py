from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def validate_ip_address_is_network_address(ip_address, module):
    """
    Validate if the given IP address is a network address (i.e. it's host bits are set to 0)
    ONTAP doesn't validate if the host bits are set,
    and hence doesn't add a new address unless the IP is from a different network.
    So this validation allows the module to be idempotent.
    :return: None
    """
    dummy = _get_ipv4orv6_network(ip_address, None, True, module)