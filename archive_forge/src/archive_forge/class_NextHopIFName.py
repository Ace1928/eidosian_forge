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
@_NextHop.register_type(ZEBRA_NEXTHOP_IFNAME)
class NextHopIFName(_NextHop):
    """
    Nexthop class for ZEBRA_NEXTHOP_IFNAME type.
    """
    _BODY_FMT = '!I'
    BODY_SIZE = struct.calcsize(_BODY_FMT)

    @classmethod
    def parse(cls, buf):
        ifindex, = struct.unpack_from(cls._BODY_FMT, buf)
        rest = buf[cls.BODY_SIZE:]
        return (cls(ifindex=ifindex), rest)

    def _serialize(self):
        return struct.pack(self._BODY_FMT, self.ifindex)