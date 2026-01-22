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
@_FrrZebraMessageBody.register_type(FRR_ZEBRA_INTERFACE_BFD_DEST_UPDATE)
class ZebraInterfaceBfdDestinationUpdate(_ZebraMessageBody):
    """
    Message body class for FRR_ZEBRA_INTERFACE_BFD_DEST_UPDATE.
    """
    _HEADER_FMT = '!I'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _STATUS_FMT = '!B'
    STATUS_SIZE = struct.calcsize(_STATUS_FMT)

    def __init__(self, ifindex, dst_family, dst_prefix, status, src_family, src_prefix):
        super(ZebraInterfaceBfdDestinationUpdate, self).__init__()
        self.ifindex = ifindex
        self.dst_family = dst_family
        if isinstance(dst_prefix, (IPv4Prefix, IPv6Prefix)):
            dst_prefix = dst_prefix.prefix
        self.dst_prefix = dst_prefix
        self.status = status
        self.src_family = src_family
        if isinstance(src_prefix, (IPv4Prefix, IPv6Prefix)):
            src_prefix = src_prefix.prefix
        self.src_prefix = src_prefix

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        ifindex, = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        dst_family, dst_prefix, rest = _parse_zebra_family_prefix(rest)
        status, = struct.unpack_from(cls._STATUS_FMT, rest)
        rest = rest[cls.STATUS_SIZE:]
        src_family, src_prefix, _ = _parse_zebra_family_prefix(rest)
        return cls(ifindex, dst_family, dst_prefix, status, src_family, src_prefix)

    def serialize(self, version=_DEFAULT_VERSION):
        self.dst_family, dst_bin = _serialize_zebra_family_prefix(self.dst_prefix)
        status_bin = struct.pack(self._STATUS_FMT, self.status)
        self.src_family, src_bin = _serialize_zebra_family_prefix(self.src_prefix)
        return struct.pack(self._HEADER_FMT, self.ifindex) + dst_bin + status_bin + src_bin