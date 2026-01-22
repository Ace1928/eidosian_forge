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
@_FrrNextHop.register_type(FRR_ZEBRA_NEXTHOP_IPV4)
@_NextHop.register_type(ZEBRA_NEXTHOP_IPV4)
class NextHopIPv4(_NextHop):
    """
    Nexthop class for ZEBRA_NEXTHOP_IPV4 type.
    """
    _BODY_FMT = '!4s'
    BODY_SIZE = struct.calcsize(_BODY_FMT)
    _BODY_FMT_FRR_V3 = '!4sI'
    BODY_SIZE_FRR_V3 = struct.calcsize(_BODY_FMT_FRR_V3)

    @classmethod
    def parse(cls, buf):
        if _is_frr_version_ge(_FRR_VERSION_3_0):
            addr, ifindex = struct.unpack_from(cls._BODY_FMT_FRR_V3, buf)
            addr = addrconv.ipv4.bin_to_text(addr)
            rest = buf[cls.BODY_SIZE_FRR_V3:]
            return (cls(ifindex=ifindex, addr=addr), rest)
        addr = addrconv.ipv4.bin_to_text(buf[:cls.BODY_SIZE])
        rest = buf[cls.BODY_SIZE:]
        return (cls(addr=addr), rest)

    def _serialize(self):
        if _is_frr_version_ge(_FRR_VERSION_3_0) and self.ifindex:
            addr = addrconv.ipv4.text_to_bin(self.addr)
            return struct.pack(self._BODY_FMT_FRR_V3, addr, self.ifindex)
        return addrconv.ipv4.text_to_bin(self.addr)