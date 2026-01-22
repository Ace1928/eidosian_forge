import xcffib
import struct
import io
from . import xproto
from . import shm
def GetPortAttribute(self, port, attribute, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', port, attribute))
    return self.send_request(14, buf, GetPortAttributeCookie, is_checked=is_checked)