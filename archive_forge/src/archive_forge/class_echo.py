import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import addrconv
from os_ken.lib import stringify
@icmpv6.register_icmpv6_type(ICMPV6_ECHO_REPLY, ICMPV6_ECHO_REQUEST)
class echo(_ICMPv6Payload):
    """ICMPv6 sub encoder/decoder class for Echo Request and Echo Reply
    messages.

    This is used with os_ken.lib.packet.icmpv6.icmpv6 for
    ICMPv6 Echo Request and Echo Reply messages.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    ============== ====================
    Attribute      Description
    ============== ====================
    id             Identifier
    seq            Sequence Number
    data           Data
    ============== ====================
    """
    _PACK_STR = '!HH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, id_=0, seq=0, data=None):
        self.id = id_
        self.seq = seq
        self.data = data

    @classmethod
    def parser(cls, buf, offset):
        id_, seq = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(id_, seq)
        offset += cls._MIN_LEN
        if len(buf) > offset:
            msg.data = buf[offset:]
        return msg

    def serialize(self):
        hdr = bytearray(struct.pack(echo._PACK_STR, self.id, self.seq))
        if self.data is not None:
            hdr += bytearray(self.data)
        return hdr

    def __len__(self):
        length = self._MIN_LEN
        if self.data is not None:
            length += len(self.data)
        return length