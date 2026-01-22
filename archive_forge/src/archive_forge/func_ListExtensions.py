import xcffib
import struct
import io
def ListExtensions(self, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    return self.send_request(99, buf, ListExtensionsCookie, is_checked=is_checked)