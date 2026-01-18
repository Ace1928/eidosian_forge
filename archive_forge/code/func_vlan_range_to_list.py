from __future__ import absolute_import, division, print_function
import socket
from itertools import count, groupby
from ansible.module_utils.common.network import is_masklen, to_netmask
from ansible.module_utils.six import iteritems
def vlan_range_to_list(vlans):
    result = []
    if vlans:
        for part in vlans:
            if part == 'none':
                break
            if '-' in part:
                a, b = part.split('-')
                a, b = (int(a), int(b))
                result.extend(range(a, b + 1))
            else:
                a = int(part)
                result.append(a)
        return numerical_sort(result)
    return result