import xcffib
import struct
import io
from . import xproto
def AllocateBackBuffer(self, window, buffer, swap_action, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIB3x', window, buffer, swap_action))
    return self.send_request(1, buf, is_checked=is_checked)