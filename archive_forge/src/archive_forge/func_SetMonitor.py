import xcffib
import struct
import io
from . import xproto
from . import render
def SetMonitor(self, window, monitorinfo, is_checked=False):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', window))
    buf.write(monitorinfo.pack() if hasattr(monitorinfo, 'pack') else MonitorInfo.synthetic(*monitorinfo).pack())
    return self.send_request(43, buf, is_checked=is_checked)