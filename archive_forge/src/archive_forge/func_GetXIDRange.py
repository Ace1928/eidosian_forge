import xcffib
import struct
import io
def GetXIDRange(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(1, buf, GetXIDRangeCookie, is_checked=is_checked)