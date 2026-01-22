import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@sctp.register_chunk_type
class chunk_error(chunk):
    """Stream Control Transmission Protocol (SCTP)
    sub encoder/decoder class for Operation Error (ERROR) chunk (RFC 4960).

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
    causes         a list of derived classes of os_ken.lib.packet.sctp.causes.
    ============== =====================================================
    """
    _class_prefixes = ['cause_']
    _RECOGNIZED_CAUSES = {}

    @staticmethod
    def register_cause_code(*args):

        def _register_cause_code(cls):
            chunk_error._RECOGNIZED_CAUSES[cls.cause_code()] = cls
            return cls
        return _register_cause_code(args[0])

    @classmethod
    def chunk_type(cls):
        return TYPE_ERROR

    def __init__(self, flags=0, length=0, causes=None):
        super(chunk_error, self).__init__(self.chunk_type(), length)
        self.flags = flags
        causes = causes or []
        assert isinstance(causes, list)
        for one in causes:
            assert isinstance(one, cause)
        self.causes = causes

    @classmethod
    def parser(cls, buf):
        _, flags, length = struct.unpack_from(cls._PACK_STR, buf)
        causes = []
        offset = cls._MIN_LEN
        while offset < length:
            ccode, = struct.unpack_from('!H', buf, offset)
            cls_ = cls._RECOGNIZED_CAUSES.get(ccode)
            if not cls_:
                break
            ins = cls_.parser(buf[offset:])
            causes.append(ins)
            offset += len(ins)
        return cls(flags, length, causes)

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length))
        for one in self.causes:
            buf.extend(one.serialize())
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        return bytes(buf)