import xcffib
import struct
import io
def FillPoly(self, drawable, gc, shape, coordinate_mode, points_len, points, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIBB2x', drawable, gc, shape, coordinate_mode))
    buf.write(xcffib.pack_list(points, POINT))
    return self.send_request(69, buf, is_checked=is_checked)