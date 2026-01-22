import xcffib
import struct
import io
def UnregisterClients(self, context, num_client_specs, client_specs, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xII', context, num_client_specs))
    buf.write(xcffib.pack_list(client_specs, 'I'))
    return self.send_request(3, buf, is_checked=is_checked)