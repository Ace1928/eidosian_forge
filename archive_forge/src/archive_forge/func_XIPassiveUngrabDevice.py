import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIPassiveUngrabDevice(self, grab_window, detail, deviceid, num_modifiers, grab_type, modifiers, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHHB3x', grab_window, detail, deviceid, num_modifiers, grab_type))
    buf.write(xcffib.pack_list(modifiers, 'I'))
    return self.send_request(55, buf, is_checked=is_checked)