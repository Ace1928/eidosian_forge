import xcffib
import struct
import io
from . import xproto
from . import render
def SetCrtcTransform(self, crtc, transform, filter_len, filter_name, filter_params_len, filter_params, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', crtc))
    buf.write(transform.pack() if hasattr(transform, 'pack') else render.TRANSFORM.synthetic(*transform).pack())
    buf.write(struct.pack('=H', filter_len))
    buf.write(struct.pack('=2x'))
    buf.write(xcffib.pack_list(filter_name, 'c'))
    buf.write(struct.pack('=4x'))
    buf.write(xcffib.pack_list(filter_params, 'i'))
    return self.send_request(26, buf, is_checked=is_checked)