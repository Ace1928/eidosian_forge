import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIQueryDevice(self, deviceid, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', deviceid))
    return self.send_request(48, buf, XIQueryDeviceCookie, is_checked=is_checked)