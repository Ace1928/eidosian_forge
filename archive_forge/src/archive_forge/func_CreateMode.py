import xcffib
import struct
import io
from . import xproto
from . import render
def CreateMode(self, window, mode_info, name_len, name, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    buf.write(mode_info.pack() if hasattr(mode_info, 'pack') else ModeInfo.synthetic(*mode_info).pack())
    buf.write(xcffib.pack_list(name, 'c'))
    return self.send_request(16, buf, CreateModeCookie, is_checked=is_checked)