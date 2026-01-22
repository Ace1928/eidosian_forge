import xcffib
import struct
import io
def GetProperty(self, delete, window, property, type, long_offset, long_length, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIIIII', delete, window, property, type, long_offset, long_length))
    return self.send_request(20, buf, GetPropertyCookie, is_checked=is_checked)