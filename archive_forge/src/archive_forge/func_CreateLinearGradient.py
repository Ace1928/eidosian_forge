import xcffib
import struct
import io
from . import xproto
def CreateLinearGradient(self, picture, p1, p2, num_stops, stops, colors, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', picture))
    buf.write(p1.pack() if hasattr(p1, 'pack') else POINTFIX.synthetic(*p1).pack())
    buf.write(p2.pack() if hasattr(p2, 'pack') else POINTFIX.synthetic(*p2).pack())
    buf.write(struct.pack('=I', num_stops))
    buf.write(xcffib.pack_list(stops, 'i'))
    buf.write(xcffib.pack_list(colors, COLOR))
    return self.send_request(34, buf, is_checked=is_checked)