import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def GetCursorImageAndName(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(25, buf, GetCursorImageAndNameCookie, is_checked=is_checked)