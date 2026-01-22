import xcffib
import struct
import io
from . import xfixes
from . import xproto
def UngrabDeviceKey(self, grabWindow, modifiers, modifier_device, key, grabbed_device, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHBBB', grabWindow, modifiers, modifier_device, key, grabbed_device))
    return self.send_request(16, buf, is_checked=is_checked)