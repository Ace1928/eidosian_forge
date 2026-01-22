import xcffib
import struct
import io
from . import xfixes
from . import xproto
def DeleteDeviceProperty(self, property, device_id, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIB3x', property, device_id))
    return self.send_request(38, buf, is_checked=is_checked)