import xcffib
import struct
import io
from . import xproto
def DeleteWindow(self, glxwindow, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', glxwindow))
    return self.send_request(32, buf, is_checked=is_checked)