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
class _ZebraIPRoute(_ZebraMessageBody, metaclass=abc.ABCMeta):
    """
    Base class for ZEBRA_IPV4_ROUTE_* and ZEBRA_IPV6_ROUTE_*
    message body.

    .. Note::

        Zebra IPv4/IPv6 Route message have asymmetric structure.
        If the message sent from Zebra Daemon, set 'from_zebra=True' to
        create an instance of this class.
    """
    _HEADER_FMT = '!BBB'
    HEADER_SIZE = struct.calcsize(_HEADER_FMT)
    _V4_HEADER_FMT = '!BHIB'
    V4_HEADER_SIZE = struct.calcsize(_V4_HEADER_FMT)
    _SAFI_FMT = '!H'
    SAFI_SIZE = struct.calcsize(_SAFI_FMT)
    _NUM_FMT = '!B'
    NUM_SIZE = struct.calcsize(_NUM_FMT)
    _IFINDEX_FMT = '!I'
    IFINDEX_SIZE = struct.calcsize(_IFINDEX_FMT)
    _FAMILY = None

    def __init__(self, route_type, flags, message, safi=None, prefix=None, src_prefix=None, nexthops=None, ifindexes=None, distance=None, metric=None, mtu=None, tag=None, instance=None, from_zebra=False):
        super(_ZebraIPRoute, self).__init__()
        self.route_type = route_type
        self.instance = instance
        self.flags = flags
        self.message = message
        if from_zebra:
            self.safi = None
        else:
            self.safi = safi or packet_safi.UNICAST
        assert prefix is not None
        if isinstance(prefix, (IPv4Prefix, IPv6Prefix)):
            prefix = prefix.prefix
        self.prefix = prefix
        if isinstance(src_prefix, (IPv4Prefix, IPv6Prefix)):
            src_prefix = src_prefix.prefix
        self.src_prefix = src_prefix
        nexthops = nexthops or []
        if from_zebra:
            for nexthop in nexthops:
                assert ip.valid_ipv4(nexthop) or ip.valid_ipv6(nexthop)
        else:
            for nexthop in nexthops:
                assert isinstance(nexthop, _NextHop)
        self.nexthops = nexthops
        if from_zebra:
            ifindexes = ifindexes or []
            for ifindex in ifindexes:
                assert isinstance(ifindex, int)
            self.ifindexes = ifindexes
        else:
            self.ifindexes = None
        self.distance = distance
        self.metric = metric
        self.mtu = mtu
        self.tag = tag
        self.from_zebra = from_zebra

    @classmethod
    def _parse_message_option(cls, message, flag, fmt, buf):
        if message & flag:
            option, = struct.unpack_from(fmt, buf)
            return (option, buf[struct.calcsize(fmt):])
        return (None, buf)

    @classmethod
    def _parse_impl(cls, buf, version=_DEFAULT_VERSION, from_zebra=False):
        instance = None
        if version <= 3:
            route_type, flags, message = struct.unpack_from(cls._HEADER_FMT, buf)
            rest = buf[cls.HEADER_SIZE:]
        elif version == 4:
            route_type, instance, flags, message = struct.unpack_from(cls._V4_HEADER_FMT, buf)
            rest = buf[cls.V4_HEADER_SIZE:]
        else:
            raise struct.error('Unsupported Zebra protocol version: %d' % version)
        if from_zebra:
            safi = None
        else:
            safi, = struct.unpack_from(cls._SAFI_FMT, rest)
            rest = rest[cls.SAFI_SIZE:]
        prefix, rest = _parse_ip_prefix(cls._FAMILY, rest)
        src_prefix = None
        if version == 4 and message & FRR_ZAPI_MESSAGE_SRCPFX:
            src_prefix, rest = _parse_ip_prefix(cls._FAMILY, rest)
        if from_zebra and message & ZAPI_MESSAGE_NEXTHOP:
            nexthops = []
            nexthop_num, = struct.unpack_from(cls._NUM_FMT, rest)
            rest = rest[cls.NUM_SIZE:]
            if cls._FAMILY == socket.AF_INET:
                for _ in range(nexthop_num):
                    nexthop = addrconv.ipv4.bin_to_text(rest[:4])
                    nexthops.append(nexthop)
                    rest = rest[4:]
            else:
                for _ in range(nexthop_num):
                    nexthop = addrconv.ipv6.bin_to_text(rest[:16])
                    nexthops.append(nexthop)
                    rest = rest[16:]
        else:
            nexthops, rest = _parse_nexthops(rest, version)
        ifindexes = []
        if from_zebra and message & ZAPI_MESSAGE_IFINDEX:
            ifindex_num, = struct.unpack_from(cls._NUM_FMT, rest)
            rest = rest[cls.NUM_SIZE:]
            for _ in range(ifindex_num):
                ifindex, = struct.unpack_from(cls._IFINDEX_FMT, rest)
                ifindexes.append(ifindex)
                rest = rest[cls.IFINDEX_SIZE:]
        if version <= 3:
            distance, rest = cls._parse_message_option(message, ZAPI_MESSAGE_DISTANCE, '!B', rest)
            metric, rest = cls._parse_message_option(message, ZAPI_MESSAGE_METRIC, '!I', rest)
            mtu, rest = cls._parse_message_option(message, ZAPI_MESSAGE_MTU, '!I', rest)
            tag, rest = cls._parse_message_option(message, ZAPI_MESSAGE_TAG, '!I', rest)
        elif version == 4:
            distance, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_DISTANCE, '!B', rest)
            metric, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_METRIC, '!I', rest)
            tag, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_TAG, '!I', rest)
            mtu, rest = cls._parse_message_option(message, FRR_ZAPI_MESSAGE_MTU, '!I', rest)
        else:
            raise struct.error('Unsupported Zebra protocol version: %d' % version)
        return cls(route_type, flags, message, safi, prefix, src_prefix, nexthops, ifindexes, distance, metric, mtu, tag, instance, from_zebra=from_zebra)

    @classmethod
    def parse(cls, buf, version=_DEFAULT_VERSION):
        return cls._parse_impl(buf, version=version)

    @classmethod
    def parse_from_zebra(cls, buf, version=_DEFAULT_VERSION):
        return cls._parse_impl(buf, version=version, from_zebra=True)

    def _serialize_message_option(self, option, flag, fmt):
        if option is None:
            return b''
        self.message |= flag
        return struct.pack(fmt, option)

    def serialize(self, version=_DEFAULT_VERSION):
        prefix = _serialize_ip_prefix(self.prefix)
        if version == 4 and self.src_prefix:
            self.message |= FRR_ZAPI_MESSAGE_SRCPFX
            prefix += _serialize_ip_prefix(self.src_prefix)
        nexthops = b''
        if self.from_zebra and self.nexthops:
            self.message |= ZAPI_MESSAGE_NEXTHOP
            nexthops += struct.pack(self._NUM_FMT, len(self.nexthops))
            for nexthop in self.nexthops:
                nexthops += ip.text_to_bin(nexthop)
        else:
            self.message |= ZAPI_MESSAGE_NEXTHOP
            nexthops = _serialize_nexthops(self.nexthops, version=version)
        ifindexes = b''
        if self.ifindexes and self.from_zebra:
            self.message |= ZAPI_MESSAGE_IFINDEX
            ifindexes += struct.pack(self._NUM_FMT, len(self.ifindexes))
            for ifindex in self.ifindexes:
                ifindexes += struct.pack(self._IFINDEX_FMT, ifindex)
        if version <= 3:
            options = self._serialize_message_option(self.distance, ZAPI_MESSAGE_DISTANCE, '!B')
            options += self._serialize_message_option(self.metric, ZAPI_MESSAGE_METRIC, '!I')
            options += self._serialize_message_option(self.mtu, ZAPI_MESSAGE_MTU, '!I')
            options += self._serialize_message_option(self.tag, ZAPI_MESSAGE_TAG, '!I')
            header = struct.pack(self._HEADER_FMT, self.route_type, self.flags, self.message)
        elif version == 4:
            options = self._serialize_message_option(self.distance, FRR_ZAPI_MESSAGE_DISTANCE, '!B')
            options += self._serialize_message_option(self.metric, FRR_ZAPI_MESSAGE_METRIC, '!I')
            options += self._serialize_message_option(self.tag, FRR_ZAPI_MESSAGE_TAG, '!I')
            options += self._serialize_message_option(self.mtu, FRR_ZAPI_MESSAGE_MTU, '!I')
            header = struct.pack(self._V4_HEADER_FMT, self.route_type, self.instance, self.flags, self.message)
        else:
            raise ValueError('Unsupported Zebra protocol version: %d' % version)
        if not self.from_zebra:
            header += struct.pack(self._SAFI_FMT, self.safi)
        return header + prefix + nexthops + ifindexes + options