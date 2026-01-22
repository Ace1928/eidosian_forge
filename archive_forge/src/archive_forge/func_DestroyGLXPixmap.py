import xcffib
import struct
import io
from . import xproto
def DestroyGLXPixmap(self, glx_pixmap, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', glx_pixmap))
    return self.send_request(15, buf, is_checked=is_checked)