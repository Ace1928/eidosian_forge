import xcffib
import struct
import io
from . import xproto
def Composite(self, op, src, mask, dst, src_x, src_y, mask_x, mask_y, dst_x, dst_y, width, height, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3xIIIhhhhhhHH', op, src, mask, dst, src_x, src_y, mask_x, mask_y, dst_x, dst_y, width, height))
    return self.send_request(8, buf, is_checked=is_checked)