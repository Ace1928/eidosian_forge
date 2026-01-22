import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XISetFocus(self, window, time, deviceid, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIH2x', window, time, deviceid))
    return self.send_request(49, buf, is_checked=is_checked)