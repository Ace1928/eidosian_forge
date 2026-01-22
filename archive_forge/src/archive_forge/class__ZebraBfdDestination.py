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
class _ZebraBfdDestination(_ZebraMessageBody):
    """
    Base class for FRR_ZEBRA_BFD_DEST_REGISTER and
    FRR_ZEBRA_BFD_DEST_UPDATE message body.
    """
    _HEADER_FMT = '!I'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _FAMILY_FMT = '!H'
    FAMILY_SIZE = struct.calcsize(_FAMILY_FMT)
    _BODY_FMT = '!IIBB'
    BODY_SIZE = struct.calcsize(_BODY_FMT)
    _FOOTER_FMT = '!B'
    FOOTER_SIZE = struct.calcsize(_FOOTER_FMT)

    def __init__(self, pid, dst_family, dst_prefix, min_rx_timer, min_tx_timer, detect_mult, multi_hop, src_family, src_prefix, multi_hop_count=None, ifname=None):
        super(_ZebraBfdDestination, self).__init__()
        self.pid = pid
        self.dst_family = dst_family
        assert ip.valid_ipv4(dst_prefix) or ip.valid_ipv6(dst_prefix)
        self.dst_prefix = dst_prefix
        self.min_rx_timer = min_rx_timer
        self.min_tx_timer = min_tx_timer
        self.detect_mult = detect_mult
        self.multi_hop = multi_hop
        self.src_family = src_family
        assert ip.valid_ipv4(src_prefix) or ip.valid_ipv6(src_prefix)
        self.src_prefix = src_prefix
        self.multi_hop_count = multi_hop_count
        self.ifname = ifname

    @classmethod
    def _parse_family_prefix(cls, buf):
        family, = struct.unpack_from(cls._FAMILY_FMT, buf)
        rest = buf[cls.FAMILY_SIZE:]
        if socket.AF_INET == family:
            return (family, addrconv.ipv4.bin_to_text(rest[:4]), rest[4:])
        elif socket.AF_INET6 == family:
            return (family, addrconv.ipv6.bin_to_text(rest[:16]), rest[16:])
        raise struct.error('Unsupported family: %d' % family)

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        pid, = struct.unpack_from(cls._HEADER_FMT, buf)
        rest = buf[cls.HEADER_SIZE:]
        dst_family, dst_prefix, rest = cls._parse_family_prefix(rest)
        min_rx_timer, min_tx_timer, detect_mult, multi_hop = struct.unpack_from(cls._BODY_FMT, rest)
        rest = rest[cls.BODY_SIZE:]
        src_family, src_prefix, rest = cls._parse_family_prefix(rest)
        multi_hop_count = None
        ifname = None
        if multi_hop:
            multi_hop_count, = struct.unpack_from(cls._FOOTER_FMT, rest)
        else:
            ifname_len, = struct.unpack_from(cls._FOOTER_FMT, rest)
            ifname_bin = rest[cls.FOOTER_SIZE:cls.FOOTER_SIZE + ifname_len]
            ifname = str(str(ifname_bin.strip(b'\x00'), 'ascii'))
        return cls(pid, dst_family, dst_prefix, min_rx_timer, min_tx_timer, detect_mult, multi_hop, src_family, src_prefix, multi_hop_count, ifname)

    def _serialize_family_prefix(self, prefix):
        if ip.valid_ipv4(prefix):
            family = socket.AF_INET
            return (family, struct.pack(self._FAMILY_FMT, family) + addrconv.ipv4.text_to_bin(prefix))
        elif ip.valid_ipv6(prefix):
            family = socket.AF_INET6
            return (family, struct.pack(self._FAMILY_FMT, family) + addrconv.ipv6.text_to_bin(prefix))
        raise ValueError('Invalid prefix: %s' % prefix)

    def serialize(self, version=_DEFAULT_VERSION):
        self.dst_family, dst_bin = self._serialize_family_prefix(self.dst_prefix)
        body_bin = struct.pack(self._BODY_FMT, self.min_rx_timer, self.min_tx_timer, self.detect_mult, self.multi_hop)
        self.src_family, src_bin = self._serialize_family_prefix(self.src_prefix)
        if self.multi_hop:
            footer_bin = struct.pack(self._FOOTER_FMT, self.multi_hop_count)
        else:
            ifname_bin = self.ifname.encode('ascii')
            footer_bin = struct.pack(self._FOOTER_FMT, len(ifname_bin)) + ifname_bin
        return struct.pack(self._HEADER_FMT, self.pid) + dst_bin + body_bin + src_bin + footer_bin