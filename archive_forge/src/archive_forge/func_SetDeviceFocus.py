import xcffib
import struct
import io
from . import xfixes
from . import xproto
def SetDeviceFocus(self, focus, time, revert_to, device_id, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIBB2x', focus, time, revert_to, device_id))
    return self.send_request(21, buf, is_checked=is_checked)