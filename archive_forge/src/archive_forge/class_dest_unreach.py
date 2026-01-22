import abc
import struct
from . import packet_base
from . import packet_utils
from os_ken.lib import stringify
@icmp.register_icmp_type(ICMP_DEST_UNREACH)
class dest_unreach(_ICMPv4Payload):
    """ICMP sub encoder/decoder class for Destination Unreachable Message.

    This is used with os_ken.lib.packet.icmp.icmp for
    ICMP Destination Unreachable Message.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    [RFC1191] reserves bits for the "Next-Hop MTU" field.
    [RFC4884] introduced 8-bit data length attribute.

    .. tabularcolumns:: |l|p{35em}|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    data_len       data length
    mtu            Next-Hop MTU

                   NOTE: This field is required when icmp code is 4

                   code 4 = fragmentation needed and DF set
    data           Internet Header + leading octets of original datagram
    ============== =====================================================
    """
    _PACK_STR = '!xBH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, data_len=0, mtu=0, data=None):
        super(dest_unreach, self).__init__()
        if data_len >= 0 and data_len <= 255:
            self.data_len = data_len
        else:
            raise ValueError('Specified data length (%d) is invalid.' % data_len)
        self.mtu = mtu
        self.data = data

    @classmethod
    def parser(cls, buf, offset):
        data_len, mtu = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(data_len, mtu)
        offset += cls._MIN_LEN
        if len(buf) > offset:
            msg.data = buf[offset:]
        return msg

    def serialize(self):
        hdr = bytearray(struct.pack(dest_unreach._PACK_STR, self.data_len, self.mtu))
        if self.data is not None:
            hdr += self.data
        return hdr

    def __len__(self):
        length = self._MIN_LEN
        if self.data is not None:
            length += len(self.data)
        return length