from __future__ import (absolute_import, division, print_function)
import re
from struct import pack
from socket import inet_ntoa
from ansible.module_utils.six.moves import zip
def to_netmask(val):
    """ converts a masklen to a netmask """
    if not is_masklen(val):
        raise ValueError('invalid value for masklen')
    bits = 0
    for i in range(32 - int(val), 32):
        bits |= 1 << i
    return inet_ntoa(pack('>I', bits))