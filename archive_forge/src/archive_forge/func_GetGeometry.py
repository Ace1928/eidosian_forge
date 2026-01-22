import xcffib
import struct
import io
def GetGeometry(self, drawable, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', drawable))
    return self.send_request(14, buf, GetGeometryCookie, is_checked=is_checked)