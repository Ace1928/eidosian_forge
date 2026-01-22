from __future__ import absolute_import, division, print_function
import binascii
import contextlib
import datetime
import errno
import math
import mmap
import os
import re
import select
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.sys_info import get_platform_subclass
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.compat.datetime import utcnow
def _convert_host_to_hex(host):
    """
    Convert the provided host to the format in /proc/net/tcp*

    /proc/net/tcp uses little-endian four byte hex for ipv4
    /proc/net/tcp6 uses little-endian per 4B word for ipv6

    Args:
        host: String with either hostname, IPv4, or IPv6 address

    Returns:
        List of tuples containing address family and the
        little-endian converted host
    """
    ips = []
    if host is not None:
        for family, ip in _convert_host_to_ip(host):
            hexip_nf = binascii.b2a_hex(socket.inet_pton(family, ip))
            hexip_hf = ''
            for i in range(0, len(hexip_nf), 8):
                ipgroup_nf = hexip_nf[i:i + 8]
                ipgroup_hf = socket.ntohl(int(ipgroup_nf, base=16))
                hexip_hf = '%s%08X' % (hexip_hf, ipgroup_hf)
            ips.append((family, hexip_hf))
    return ips