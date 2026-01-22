import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import stringify
class icmp(packet_base.PacketBase):
    """ICMP (RFC 792) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== ====================
    Attribute      Description
    ============== ====================
    type           Type
    code           Code
    csum           CheckSum                    (0 means automatically-calculate when encoding)
    data           Payload.                    Either a bytearray, or                    os_ken.lib.packet.icmp.echo or                    os_ken.lib.packet.icmp.dest_unreach or                    os_ken.lib.packet.icmp.TimeExceeded object                    NOTE for icmp.echo:                    This includes "unused" 16 bits and the following                    "Internet Header + 64 bits of Original Data Datagram" of                    the ICMP header.                    NOTE for icmp.dest_unreach and icmp.TimeExceeded:                    This includes "unused" 8 or 24 bits and the following                    "Internet Header + leading octets of original datagram"                    of the original packet.
    ============== ====================
    """
    _PACK_STR = '!BBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _ICMP_TYPES = {}

    @staticmethod
    def register_icmp_type(*args):

        def _register_icmp_type(cls):
            for type_ in args:
                icmp._ICMP_TYPES[type_] = cls
            return cls
        return _register_icmp_type

    def __init__(self, type_=ICMP_ECHO_REQUEST, code=0, csum=0, data=b''):
        super(icmp, self).__init__()
        self.type = type_
        self.code = code
        self.csum = csum
        self.data = data

    @classmethod
    def parser(cls, buf):
        type_, code, csum = struct.unpack_from(cls._PACK_STR, buf)
        msg = cls(type_, code, csum)
        offset = cls._MIN_LEN
        if len(buf) > offset:
            cls_ = cls._ICMP_TYPES.get(type_, None)
            if cls_:
                msg.data = cls_.parser(buf, offset)
            else:
                msg.data = buf[offset:]
        return (msg, None, None)

    def serialize(self, payload, prev):
        hdr = bytearray(struct.pack(icmp._PACK_STR, self.type, self.code, self.csum))
        if self.data:
            if self.type in icmp._ICMP_TYPES:
                assert isinstance(self.data, _ICMPv4Payload)
                hdr += self.data.serialize()
            else:
                hdr += self.data
        else:
            self.data = echo()
            hdr += self.data.serialize()
        if self.csum == 0:
            self.csum = packet_utils.checksum(hdr)
            struct.pack_into('!H', hdr, 2, self.csum)
        return hdr

    def __len__(self):
        return self._MIN_LEN + len(self.data)