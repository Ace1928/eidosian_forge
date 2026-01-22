import xcffib
import struct
import io
from . import xproto
def CreateCounter(self, id, initial_value, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', id))
    buf.write(initial_value.pack() if hasattr(initial_value, 'pack') else INT64.synthetic(*initial_value).pack())
    return self.send_request(2, buf, is_checked=is_checked)