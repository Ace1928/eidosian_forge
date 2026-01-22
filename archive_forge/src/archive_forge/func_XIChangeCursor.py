import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIChangeCursor(self, window, cursor, deviceid, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIH2x', window, cursor, deviceid))
    return self.send_request(42, buf, is_checked=is_checked)