import xcffib
import struct
import io
class SetupFailed(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.status, self.reason_len, self.protocol_major_version, self.protocol_minor_version, self.length = unpacker.unpack('BBHHH')
        self.reason = xcffib.List(unpacker, 'c', self.reason_len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BBHHH', self.status, self.reason_len, self.protocol_major_version, self.protocol_minor_version, self.length))
        buf.write(xcffib.pack_list(self.reason, 'c'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, status, reason_len, protocol_major_version, protocol_minor_version, length, reason):
        self = cls.__new__(cls)
        self.status = status
        self.reason_len = reason_len
        self.protocol_major_version = protocol_major_version
        self.protocol_minor_version = protocol_minor_version
        self.length = length
        self.reason = reason
        return self