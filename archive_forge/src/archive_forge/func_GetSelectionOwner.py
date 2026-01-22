import xcffib
import struct
import io
def GetSelectionOwner(self, selection, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', selection))
    return self.send_request(23, buf, GetSelectionOwnerCookie, is_checked=is_checked)