import xcffib
import struct
import io
from . import xfixes
from . import xproto
def SetDeviceMode(self, device_id, mode, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBB2x', device_id, mode))
    return self.send_request(5, buf, SetDeviceModeCookie, is_checked=is_checked)