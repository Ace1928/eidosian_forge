import xcffib
import struct
import io
def KillClient(self, resource, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', resource))
    return self.send_request(113, buf, is_checked=is_checked)