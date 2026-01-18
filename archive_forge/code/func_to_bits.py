from __future__ import (absolute_import, division, print_function)
import re
from struct import pack
from socket import inet_ntoa
from ansible.module_utils.six.moves import zip
def to_bits(val):
    """ converts a netmask to bits """
    bits = ''
    for octet in val.split('.'):
        bits += bin(int(octet))[2:].zfill(8)
    return bits