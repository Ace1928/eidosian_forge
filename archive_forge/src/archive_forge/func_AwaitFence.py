import xcffib
import struct
import io
from . import xproto
def AwaitFence(self, fence_list_len, fence_list, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2x'))
    buf.write(xcffib.pack_list(fence_list, 'I'))
    return self.send_request(19, buf, is_checked=is_checked)