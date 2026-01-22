import xcffib
import struct
import io
from . import xproto
def CompositeGlyphs8(self, op, src, dst, mask_format, glyphset, src_x, src_y, glyphcmds_len, glyphcmds, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3xIIIIhh', op, src, dst, mask_format, glyphset, src_x, src_y))
    buf.write(xcffib.pack_list(glyphcmds, 'B'))
    return self.send_request(23, buf, is_checked=is_checked)