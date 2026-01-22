import xcffib
import struct
import io
def SetAccessControl(self, mode, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2x', mode))
    return self.send_request(111, buf, is_checked=is_checked)