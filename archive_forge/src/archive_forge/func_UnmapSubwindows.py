import xcffib
import struct
import io
def UnmapSubwindows(self, window, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    return self.send_request(11, buf, is_checked=is_checked)