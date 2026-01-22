import abc
import socket
import struct
import logging
import netaddr
from packaging import version as packaging_version
from os_ken import flags as cfg_flags  # For loading 'zapi' option definition
from os_ken.cfg import CONF
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from . import packet_base
from . import bgp
from . import safi as packet_safi
def _serialize_ip_prefix(prefix):
    if ip.valid_ipv4(prefix):
        prefix_addr, prefix_num = prefix.split('/')
        return bgp.IPAddrPrefix(int(prefix_num), prefix_addr).serialize()
    elif ip.valid_ipv6(prefix):
        prefix_addr, prefix_num = prefix.split('/')
        return IPv6Prefix(int(prefix_num), prefix_addr).serialize()
    else:
        raise ValueError('Invalid prefix: %s' % prefix)