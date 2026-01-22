import xcffib
import struct
import io
from . import xproto
from . import render
from . import shape
def CreatePointerBarrier(self, barrier, window, x1, y1, x2, y2, directions, num_devices, devices, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIHHHHI2xH', barrier, window, x1, y1, x2, y2, directions, num_devices))
    buf.write(xcffib.pack_list(devices, 'H'))
    return self.send_request(31, buf, is_checked=is_checked)