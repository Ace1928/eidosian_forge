import xcffib
import struct
import io
class CHARINFO(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.left_side_bearing, self.right_side_bearing, self.character_width, self.ascent, self.descent, self.attributes = unpacker.unpack('hhhhhH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=hhhhhH', self.left_side_bearing, self.right_side_bearing, self.character_width, self.ascent, self.descent, self.attributes))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, left_side_bearing, right_side_bearing, character_width, ascent, descent, attributes):
        self = cls.__new__(cls)
        self.left_side_bearing = left_side_bearing
        self.right_side_bearing = right_side_bearing
        self.character_width = character_width
        self.ascent = ascent
        self.descent = descent
        self.attributes = attributes
        return self