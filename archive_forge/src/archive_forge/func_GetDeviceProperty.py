import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetDeviceProperty(self, property, type, offset, len, device_id, delete, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIBB2x', property, type, offset, len, device_id, delete))
    return self.send_request(39, buf, GetDevicePropertyCookie, is_checked=is_checked)