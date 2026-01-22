import xcffib
import struct
import io
from . import xproto
from . import render
def DestroyMode(self, mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', mode))
    return self.send_request(17, buf, is_checked=is_checked)