import xcffib
import struct
import io
class QueryFontReply(xcffib.Reply):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Reply.__init__(self, unpacker)
        base = unpacker.offset
        unpacker.unpack('xx2x4x')
        self.min_bounds = CHARINFO(unpacker)
        unpacker.unpack('4x')
        unpacker.pad(CHARINFO)
        self.max_bounds = CHARINFO(unpacker)
        self.min_char_or_byte2, self.max_char_or_byte2, self.default_char, self.properties_len, self.draw_direction, self.min_byte1, self.max_byte1, self.all_chars_exist, self.font_ascent, self.font_descent, self.char_infos_len = unpacker.unpack('4xHHHHBBBBhhI')
        unpacker.pad(FONTPROP)
        self.properties = xcffib.List(unpacker, FONTPROP, self.properties_len)
        unpacker.pad(CHARINFO)
        self.char_infos = xcffib.List(unpacker, CHARINFO, self.char_infos_len)
        self.bufsize = unpacker.offset - base