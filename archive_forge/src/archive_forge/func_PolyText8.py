import xcffib
import struct
import io
def PolyText8(self, drawable, gc, x, y, items_len, items, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIhh', drawable, gc, x, y))
    buf.write(xcffib.pack_list(items, 'B'))
    return self.send_request(74, buf, is_checked=is_checked)