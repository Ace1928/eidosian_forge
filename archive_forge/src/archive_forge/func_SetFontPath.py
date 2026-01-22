import xcffib
import struct
import io
def SetFontPath(self, font_qty, font, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xH2x', font_qty))
    buf.write(xcffib.pack_list(font, STR))
    return self.send_request(51, buf, is_checked=is_checked)