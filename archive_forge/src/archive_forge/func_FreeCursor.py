import xcffib
import struct
import io
def FreeCursor(self, cursor, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', cursor))
    return self.send_request(95, buf, is_checked=is_checked)