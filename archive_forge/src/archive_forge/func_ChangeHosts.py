import xcffib
import struct
import io
def ChangeHosts(self, mode, family, address_len, address, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xBxH', mode, family, address_len))
    buf.write(xcffib.pack_list(address, 'B'))
    return self.send_request(109, buf, is_checked=is_checked)