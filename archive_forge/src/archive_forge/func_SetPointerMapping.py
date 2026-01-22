import xcffib
import struct
import io
def SetPointerMapping(self, map_len, map, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2x', map_len))
    buf.write(xcffib.pack_list(map, 'B'))
    return self.send_request(116, buf, SetPointerMappingCookie, is_checked=is_checked)