import xcffib
import struct
import io
from . import xproto
def QueryAlarm(self, alarm, is_checked=True):
    buf = io.BytesIO()
    buf.write(struct.pack('=xx2xI', alarm))
    return self.send_request(10, buf, QueryAlarmCookie, is_checked=is_checked)