import xcffib
import struct
import io
from . import xfixes
from . import xproto
def GrabDevice(self, grab_window, time, num_classes, this_device_mode, other_device_mode, owner_events, device_id, classes, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHBBBB2x', grab_window, time, num_classes, this_device_mode, other_device_mode, owner_events, device_id))
    buf.write(xcffib.pack_list(classes, 'I'))
    return self.send_request(13, buf, GrabDeviceCookie, is_checked=is_checked)