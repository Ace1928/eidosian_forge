import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIUngrabDevice(self, time, deviceid, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', time, deviceid))
    return self.send_request(52, buf, is_checked=is_checked)