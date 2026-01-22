import xcffib
import struct
import io
def GetPointerControl(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(106, buf, GetPointerControlCookie, is_checked=is_checked)