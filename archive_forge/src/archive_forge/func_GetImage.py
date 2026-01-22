import xcffib
import struct
import io
def GetImage(self, format, drawable, x, y, width, height, plane_mask, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIhhHHI', format, drawable, x, y, width, height, plane_mask))
    return self.send_request(73, buf, GetImageCookie, is_checked=is_checked)