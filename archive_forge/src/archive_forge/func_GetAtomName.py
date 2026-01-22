import xcffib
import struct
import io
def GetAtomName(self, atom, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', atom))
    return self.send_request(17, buf, GetAtomNameCookie, is_checked=is_checked)