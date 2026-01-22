import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIPassiveGrabDevice(self, time, grab_window, cursor, detail, deviceid, num_modifiers, mask_len, grab_type, grab_mode, paired_device_mode, owner_events, mask, modifiers, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIIHHHBBBB2x', time, grab_window, cursor, detail, deviceid, num_modifiers, mask_len, grab_type, grab_mode, paired_device_mode, owner_events))
    buf.write(xcffib.pack_list(mask, 'I'))
    buf.write(xcffib.pack_list(modifiers, 'I'))
    return self.send_request(54, buf, XIPassiveGrabDeviceCookie, is_checked=is_checked)