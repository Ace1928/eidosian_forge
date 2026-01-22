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
class ZebraMessage(packet_base.PacketBase):
    """
    Zebra protocol parser/serializer class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ==========================================================
    Attribute      Description
    ============== ==========================================================
    length         Total packet length including this header.
                   The minimum length is 3 bytes for version 0 messages,
                   6 bytes for version 1/2 messages and 8 bytes for version
                   3 messages.
    version        Version number of the Zebra protocol message.
                   To instantiate messages with other than the default
                   version, ``version`` must be specified.
    vrf_id         VRF ID for the route contained in message.
                   Not present in version 0/1/2 messages in the on-wire
                   structure, and always 0 for theses version.
    command        Zebra Protocol command, which denotes message type.
    body           Messages body.
                   An instance of subclass of ``_ZebraMessageBody`` named
                   like "Zebra + <message name>" (e.g., ``ZebraHello``).
                   Or ``None`` if message does not contain any body.
    ============== ==========================================================

    .. Note::

        To instantiate Zebra messages, ``command`` can be omitted when the
        valid ``body`` is specified.

        ::

            >>> from os_ken.lib.packet import zebra
            >>> zebra.ZebraMessage(body=zebra.ZebraHello())
            ZebraMessage(body=ZebraHello(route_type=14),command=23,
            length=None,version=3,vrf_id=0)

        On the other hand, if ``body`` is omitted, ``command`` must be
        specified.

        ::

            >>> zebra.ZebraMessage(command=zebra.ZEBRA_INTERFACE_ADD)
            ZebraMessage(body=None,command=1,length=None,version=3,vrf_id=0)
    """
    _V0_HEADER_FMT = '!HB'
    V0_HEADER_SIZE = struct.calcsize(_V0_HEADER_FMT)
    _MIN_LEN = V0_HEADER_SIZE
    _V1_HEADER_FMT = '!HBBH'
    V1_HEADER_SIZE = struct.calcsize(_V1_HEADER_FMT)
    _V3_HEADER_FMT = '!HBBHH'
    V3_HEADER_SIZE = struct.calcsize(_V3_HEADER_FMT)
    _MARKER = 255
    _LT_MARKER = 254

    def __init__(self, length=None, version=_DEFAULT_VERSION, vrf_id=0, command=None, body=None):
        super(ZebraMessage, self).__init__()
        self.length = length
        self.version = version
        self.vrf_id = vrf_id
        self.command = command
        self.body = body

    def _fill_command(self):
        assert isinstance(self.body, _ZebraMessageBody)
        body_base_cls = _ZebraMessageBody
        if self.version == 4:
            body_base_cls = _FrrZebraMessageBody
        self.command = body_base_cls.rev_lookup_command(self.body.__class__)

    @classmethod
    def get_header_size(cls, version):
        if version == 0:
            return cls.V0_HEADER_SIZE
        elif version in [1, 2]:
            return cls.V1_HEADER_SIZE
        elif version in [3, 4]:
            return cls.V3_HEADER_SIZE
        else:
            raise ValueError('Unsupported Zebra protocol version: %d' % version)

    @classmethod
    def parse_header(cls, buf):
        length, marker = struct.unpack_from(cls._V0_HEADER_FMT, buf)
        if marker not in [cls._MARKER, cls._LT_MARKER]:
            command = marker
            body_buf = buf[cls.V0_HEADER_SIZE:length]
            return (length, 0, 0, command, body_buf)
        length, marker, version, command = struct.unpack_from(cls._V1_HEADER_FMT, buf)
        if version in [1, 2]:
            body_buf = buf[cls.V1_HEADER_SIZE:length]
            return (length, version, 0, command, body_buf)
        length, marker, version, vrf_id, command = struct.unpack_from(cls._V3_HEADER_FMT, buf)
        if version == 3 or (version == 4 and marker == cls._LT_MARKER):
            body_buf = buf[cls.V3_HEADER_SIZE:length]
            return (length, version, vrf_id, command, body_buf)
        raise struct.error('Failed to parse Zebra protocol header: marker=%d, version=%d' % (marker, version))

    @classmethod
    def get_body_class(cls, version, command):
        if version == 4:
            return _FrrZebraMessageBody.lookup_command(command)
        else:
            return _ZebraMessageBody.lookup_command(command)

    @classmethod
    def _parser_impl(cls, buf, from_zebra=False):
        buf = bytes(buf)
        length, version, vrf_id, command, body_buf = cls.parse_header(buf)
        if body_buf:
            body_cls = cls.get_body_class(version, command)
            if from_zebra:
                body = body_cls.parse_from_zebra(body_buf, version=version)
            else:
                body = body_cls.parse(body_buf, version=version)
        else:
            body = None
        rest = buf[length:]
        if from_zebra:
            return (cls(length, version, vrf_id, command, body), _ZebraMessageFromZebra, rest)
        return (cls(length, version, vrf_id, command, body), cls, rest)

    @classmethod
    def parser(cls, buf):
        return cls._parser_impl(buf)

    def serialize_header(self, body_len):
        if self.version == 0:
            self.length = self.V0_HEADER_SIZE + body_len
            return struct.pack(self._V0_HEADER_FMT, self.length, self.command)
        elif self.version in [1, 2]:
            self.length = self.V1_HEADER_SIZE + body_len
            return struct.pack(self._V1_HEADER_FMT, self.length, self._MARKER, self.version, self.command)
        elif self.version in [3, 4]:
            if self.version == 3:
                _marker = self._MARKER
            else:
                _marker = self._LT_MARKER
            self.length = self.V3_HEADER_SIZE + body_len
            return struct.pack(self._V3_HEADER_FMT, self.length, _marker, self.version, self.vrf_id, self.command)
        else:
            raise ValueError('Unsupported Zebra protocol version: %d' % self.version)

    def serialize(self, _payload=None, _prev=None):
        if self.body is None:
            assert self.command is not None
            body = b''
        else:
            assert isinstance(self.body, _ZebraMessageBody)
            self._fill_command()
            body = self.body.serialize(version=self.version)
        return self.serialize_header(len(body)) + body