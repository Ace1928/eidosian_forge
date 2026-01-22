import xcffib
import struct
import io
def ChangeActivePointerGrab(self, cursor, time, event_mask, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIH2x', cursor, time, event_mask))
    return self.send_request(30, buf, is_checked=is_checked)