import xcffib
import struct
import io
def SendEvent(self, propagate, destination, event_mask, event, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xII', propagate, destination, event_mask))
    buf.write(xcffib.pack_list(event, 'c'))
    return self.send_request(25, buf, is_checked=is_checked)