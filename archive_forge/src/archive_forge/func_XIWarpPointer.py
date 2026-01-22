import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIWarpPointer(self, src_win, dst_win, src_x, src_y, src_width, src_height, dst_x, dst_y, deviceid, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIiiHHiiH2x', src_win, dst_win, src_x, src_y, src_width, src_height, dst_x, dst_y, deviceid))
    return self.send_request(41, buf, is_checked=is_checked)