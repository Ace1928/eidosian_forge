import xcffib
import struct
import io
from . import xproto
from . import render
def ChangeProviderProperty(self, provider, property, type, format, mode, num_items, data, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIBB2xI', provider, property, type, format, mode, num_items))
    buf.write(xcffib.pack_list(data, 'c'))
    return self.send_request(39, buf, is_checked=is_checked)