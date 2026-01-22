import xcffib
import struct
import io
from . import xproto
from . import xfixes
def ReleaseOverlayWindow(self, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(8, buf, is_checked=is_checked)