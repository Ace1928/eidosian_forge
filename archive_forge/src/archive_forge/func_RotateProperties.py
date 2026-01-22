import xcffib
import struct
import io
def RotateProperties(self, window, atoms_len, delta, atoms, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xIHh', window, atoms_len, delta))
    buf.write(xcffib.pack_list(atoms, 'I'))
    return self.send_request(114, buf, is_checked=is_checked)