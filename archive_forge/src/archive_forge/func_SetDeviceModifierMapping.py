import xcffib
import struct
import io
from . import xfixes
from . import xproto
def SetDeviceModifierMapping(self, device_id, keycodes_per_modifier, keymaps, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBB2x', device_id, keycodes_per_modifier))
    buf.write(xcffib.pack_list(keymaps, 'B'))
    return self.send_request(27, buf, SetDeviceModifierMappingCookie, is_checked=is_checked)