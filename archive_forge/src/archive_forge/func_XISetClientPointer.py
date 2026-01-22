import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XISetClientPointer(self, window, deviceid, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', window, deviceid))
    return self.send_request(44, buf, is_checked=is_checked)