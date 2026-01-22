import xcffib
import struct
import io
from . import xproto
def ClientInfo(self, major_version, minor_version, str_len, string, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIII', major_version, minor_version, str_len))
    buf.write(xcffib.pack_list(string, 'c'))
    return self.send_request(20, buf, is_checked=is_checked)