import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIQueryPointer(self, window, deviceid, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', window, deviceid))
    return self.send_request(40, buf, XIQueryPointerCookie, is_checked=is_checked)