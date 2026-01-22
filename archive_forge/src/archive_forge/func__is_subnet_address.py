import os
import socket
import struct
from six.moves.urllib.parse import urlparse
def _is_subnet_address(hostname):
    try:
        addr, netmask = hostname.split('/')
        return _is_ip_address(addr) and 0 <= int(netmask) < 32
    except ValueError:
        return False