from __future__ import (absolute_import, division, print_function)
import re
from struct import pack
from socket import inet_ntoa
from ansible.module_utils.six.moves import zip
def to_subnet(addr, mask, dotted_notation=False):
    """ coverts an addr / mask pair to a subnet in cidr notation """
    try:
        if not is_masklen(mask):
            raise ValueError
        cidr = int(mask)
        mask = to_netmask(mask)
    except ValueError:
        cidr = to_masklen(mask)
    addr = addr.split('.')
    mask = mask.split('.')
    network = list()
    for s_addr, s_mask in zip(addr, mask):
        network.append(str(int(s_addr) & int(s_mask)))
    if dotted_notation:
        return '%s %s' % ('.'.join(network), to_netmask(cidr))
    return '%s/%s' % ('.'.join(network), cidr)