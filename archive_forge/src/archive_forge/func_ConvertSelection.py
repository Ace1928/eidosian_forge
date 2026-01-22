import xcffib
import struct
import io
def ConvertSelection(self, requestor, selection, target, property, time, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIII', requestor, selection, target, property, time))
    return self.send_request(24, buf, is_checked=is_checked)