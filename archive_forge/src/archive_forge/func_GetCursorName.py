import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def GetCursorName(self, cursor, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', cursor))
    return self.send_request(24, buf, GetCursorNameCookie, is_checked=is_checked)