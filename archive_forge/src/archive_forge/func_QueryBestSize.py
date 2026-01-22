import xcffib
import struct
import io
def QueryBestSize(self, _class, drawable, width, height, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIHH', _class, drawable, width, height))
    return self.send_request(97, buf, QueryBestSizeCookie, is_checked=is_checked)