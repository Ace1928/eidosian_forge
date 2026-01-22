import xcffib
import struct
import io
class ExtRange(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.major = Range8(unpacker)
        unpacker.pad(Range16)
        self.minor = Range16(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.major.pack() if hasattr(self.major, 'pack') else Range8.synthetic(*self.major).pack())
        buf.write(self.minor.pack() if hasattr(self.minor, 'pack') else Range16.synthetic(*self.minor).pack())
        return buf.getvalue()

    @classmethod
    def synthetic(cls, major, minor):
        self = cls.__new__(cls)
        self.major = major
        self.minor = minor
        return self