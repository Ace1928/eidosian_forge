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
def _parse_ip_prefix(family, buf):
    if family == socket.AF_INET:
        prefix, rest = bgp.IPAddrPrefix.parser(buf)
    elif family == socket.AF_INET6:
        prefix, rest = IPv6Prefix.parser(buf)
    else:
        raise struct.error('Unsupported family: %d' % family)
    return (prefix.prefix, rest)