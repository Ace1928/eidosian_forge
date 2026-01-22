import xcffib
import struct
import io
def OpenFont(self, fid, name_len, name, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIH2x', fid, name_len))
    buf.write(xcffib.pack_list(name, 'c'))
    return self.send_request(45, buf, is_checked=is_checked)