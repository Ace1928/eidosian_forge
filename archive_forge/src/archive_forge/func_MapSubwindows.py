import xcffib
import struct
import io
def MapSubwindows(self, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(9, buf, is_checked=is_checked)