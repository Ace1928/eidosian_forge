import xcffib
import struct
import io
from . import xproto
def FakeInput(self, type, detail, time, root, rootX, rootY, deviceid, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xBB2xII8xhh7xB', type, detail, time, root, rootX, rootY, deviceid))
    return self.send_request(2, buf, is_checked=is_checked)