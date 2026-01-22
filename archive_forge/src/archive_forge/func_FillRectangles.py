import xcffib
import struct
import io
from . import xproto
def FillRectangles(self, op, dst, color, rects_len, rects, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xB3xI', op, dst))
    buf.write(color.pack() if hasattr(color, 'pack') else COLOR.synthetic(*color).pack())
    buf.write(xcffib.pack_list(rects, xproto.RECTANGLE))
    return self.send_request(26, buf, is_checked=is_checked)