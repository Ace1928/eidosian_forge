from __future__ import (absolute_import, division, print_function)
import re
from struct import pack
from socket import inet_ntoa
from ansible.module_utils.six.moves import zip
def to_ipv6_network(addr):
    """ IPv6 addresses are eight groupings. The first three groupings (48 bits) comprise the network address. """
    ipv6_prefix = addr.split('::')[0]
    found_groups = []
    for group in ipv6_prefix.split(':'):
        found_groups.append(group)
        if len(found_groups) == 3:
            break
    if len(found_groups) < 3:
        found_groups.append('::')
    network_addr = ''
    for group in found_groups:
        if group != '::':
            network_addr += str(group)
        network_addr += str(':')
    if not network_addr.endswith('::'):
        network_addr += str(':')
    return network_addr