import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_shutdown(chunk):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Shutdown Association (SHUTDOWN) chunk
    (RFC 4960).

    This class is used with the following.

    - os_ken.lib.packet.sctp.sctp

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =====================================================
    Attribute      Description
    ============== =====================================================
    flags          set to '0'. this field will be ignored.
    length         length of this chunk containing this header.
                   (0 means automatically-calculate when encoding)
    tsn_ack        TSN of the last DATA chunk received in sequence
                   before a gap.
    ============== =====================================================
    """
    _PACK_STR = '!BBHI'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    @classmethod
    def chunk_type(cls):
        return TYPE_SHUTDOWN

    def __init__(self, flags=0, length=0, tsn_ack=0):
        super(chunk_shutdown, self).__init__(self.chunk_type(), length)
        self.flags = flags
        self.tsn_ack = tsn_ack

    @classmethod
    def parser(cls, buf):
        _, flags, length, tsn_ack = struct.unpack_from(cls._PACK_STR, buf)
        msg = cls(flags, length, tsn_ack)
        return msg

    def serialize(self):
        if 0 == self.length:
            self.length = self._MIN_LEN
        buf = struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length, self.tsn_ack)
        return buf