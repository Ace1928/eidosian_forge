import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GetDeviceModifierMapping(self, device_id, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3x', device_id))
    return self.send_request(26, buf, GetDeviceModifierMappingCookie, is_checked=is_checked)