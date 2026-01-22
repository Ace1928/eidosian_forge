import xcffib
import struct
import io
def AllocColorPlanes(self, contiguous, cmap, colors, reds, greens, blues, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xB2xIHHHH', contiguous, cmap, colors, reds, greens, blues))
    return self.send_request(87, buf, AllocColorPlanesCookie, is_checked=is_checked)