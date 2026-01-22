import xcffib
import struct
import io
from . import xfixes
from . import xproto
def XIBarrierReleasePointer(self, num_barriers, barriers, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', num_barriers))
    buf.write(xcffib.pack_list(barriers, BarrierReleasePointerInfo))
    return self.send_request(61, buf, is_checked=is_checked)