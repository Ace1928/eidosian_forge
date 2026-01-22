import xcffib
import struct
import io
from . import xproto
def CompareCursor(self, window, cursor, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', window, cursor))
    return self.send_request(1, buf, CompareCursorCookie, is_checked=is_checked)