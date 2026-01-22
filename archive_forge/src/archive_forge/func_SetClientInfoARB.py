import xcffib
import struct
import io
from . import xproto
def SetClientInfoARB(self, major_version, minor_version, num_versions, gl_str_len, glx_str_len, gl_versions, gl_extension_string, glx_extension_string, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIIIII', major_version, minor_version, num_versions, gl_str_len, glx_str_len))
    buf.write(xcffib.pack_list(gl_versions, 'I'))
    buf.write(xcffib.pack_list(gl_extension_string, 'c'))
    buf.write(struct.pack('=4x'))
    buf.write(xcffib.pack_list(glx_extension_string, 'c'))
    return self.send_request(33, buf, is_checked=is_checked)