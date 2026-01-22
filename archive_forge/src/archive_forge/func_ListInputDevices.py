import xcffib
import struct
import io
from . import xfixes
from . import xproto
def ListInputDevices(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(2, buf, ListInputDevicesCookie, is_checked=is_checked)