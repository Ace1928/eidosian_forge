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
class _ZebraInterfaceAddress(_ZebraMessageBody, metaclass=abc.ABCMeta):
    """
    Base class for ZEBRA_INTERFACE_ADDRESS_ADD and
    ZEBRA_INTERFACE_ADDRESS_DELETE message body.
    """
    _HEADER_FMT = '!IB'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    def __init__(self, ifindex, ifc_flags, family, prefix, dest):
        super(_ZebraInterfaceAddress, self).__init__()
        self.ifindex = ifindex
        self.ifc_flags = ifc_flags
        self.family = family
        if isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        self.prefix = prefix
        assert ip.valid_ipv4(dest) or ip.valid_ipv6(dest)
        self.dest = dest

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        ifindex, ifc_flags = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        family, prefix, rest = _parse_zebra_family_prefix(rest)
        if socket.AF_INET == family:
            dest = addrconv.ipv4.bin_to_text(rest)
        elif socket.AF_INET6 == family:
            dest = addrconv.ipv6.bin_to_text(rest)
        else:
            raise struct.error('Unsupported family: %d' % family)
        return cls(ifindex, ifc_flags, family, prefix, dest)

    def serialize(self, version=_DEFAULT_VERSION):
        self.family, body_bin = _serialize_zebra_family_prefix(self.prefix)
        if ip.valid_ipv4(self.dest):
            body_bin += addrconv.ipv4.text_to_bin(self.dest)
        elif ip.valid_ipv6(self.prefix):
            body_bin += addrconv.ipv6.text_to_bin(self.dest)
        else:
            raise ValueError('Invalid destination address: %s' % self.dest)
        return struct.pack(self._HEADER_FMT, self.ifindex, self.ifc_flags) + body_bin