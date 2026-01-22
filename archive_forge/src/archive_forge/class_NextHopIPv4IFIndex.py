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
@_FrrNextHop.register_type(FRR_ZEBRA_NEXTHOP_IPV4_IFINDEX)
@_NextHop.register_type(ZEBRA_NEXTHOP_IPV4_IFINDEX)
class NextHopIPv4IFIndex(_NextHop):
    """
    Nexthop class for ZEBRA_NEXTHOP_IPV4_IFINDEX type.
    """
    _BODY_FMT = '!4sI'
    BODY_SIZE = struct.calcsize(_BODY_FMT)

    @classmethod
    def parse(cls, buf):
        addr, ifindex = struct.unpack_from(cls._BODY_FMT, buf)
        addr = addrconv.ipv4.bin_to_text(addr)
        rest = buf[cls.BODY_SIZE:]
        return (cls(ifindex=ifindex, addr=addr), rest)

    def _serialize(self):
        addr = addrconv.ipv4.text_to_bin(self.addr)
        return struct.pack(self._BODY_FMT, addr, self.ifindex)