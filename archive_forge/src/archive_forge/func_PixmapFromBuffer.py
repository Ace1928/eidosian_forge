import xcffib
import struct
import io
from . import xproto
def PixmapFromBuffer(self, pixmap, drawable, size, width, height, stride, depth, bpp, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIHHHBB', pixmap, drawable, size, width, height, stride, depth, bpp))
    return self.send_request(2, buf, is_checked=is_checked)