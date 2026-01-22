import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetSelectedExtensionEvents(self, window, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(7, buf, GetSelectedExtensionEventsCookie, is_checked=is_checked)