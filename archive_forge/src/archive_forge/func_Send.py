import xcffib
import struct
import io
def Send(self, event, data_type, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    buf.write(event.pack() if hasattr(event, 'pack') else Event.synthetic(*event).pack())
    buf.write(struct.pack('=I', data_type))
    buf.write(struct.pack('=64x'))
    return self.send_request(3, buf, SendCookie, is_checked=is_checked)