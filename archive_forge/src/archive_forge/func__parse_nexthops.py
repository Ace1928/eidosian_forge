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
def _parse_nexthops(buf, version=_DEFAULT_VERSION):
    nexthop_count, = struct.unpack_from(_NEXTHOP_COUNT_FMT, buf)
    rest = buf[_NEXTHOP_COUNT_SIZE:]
    if version <= 3:
        nh_cls = _NextHop
    elif version == 4:
        nh_cls = _FrrNextHop
    else:
        raise struct.error('Unsupported Zebra protocol version: %d' % version)
    nexthops = []
    for _ in range(nexthop_count):
        nexthop, rest = nh_cls.parse(rest)
        nexthops.append(nexthop)
    return (nexthops, rest)