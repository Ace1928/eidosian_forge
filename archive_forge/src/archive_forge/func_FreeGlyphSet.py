import xcffib
import struct
import io
from . import xproto
def FreeGlyphSet(self, glyphset, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', glyphset))
    return self.send_request(19, buf, is_checked=is_checked)