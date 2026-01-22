import xcffib
import struct
import io
from . import xproto
from . import shm
def SetPortAttribute(self, port, attribute, value, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIi', port, attribute, value))
    return self.send_request(13, buf, is_checked=is_checked)