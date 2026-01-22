from __future__ import absolute_import
import array
import struct
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
def _PutVarInt(self, value):
    if value is None:
        self.buf.append(0)
        return
    if value >= _MIN_INLINE and value <= _MAX_INLINE:
        value = _OFFSET + (value - _MIN_INLINE)
        self.buf.append(value & 255)
        return
    negative = False
    if value < 0:
        value = _MIN_INLINE - value
        negative = True
    else:
        value = value - _MAX_INLINE
    len = 0
    w = value
    while w > 0:
        w >>= 8
        len += 1
    if negative:
        head = _OFFSET - len
    else:
        head = _POS_OFFSET + len
    self.buf.append(head & 255)
    for i in range(len - 1, -1, -1):
        b = value >> i * 8
        if negative:
            b = _MAX_UNSIGNED_BYTE - (b & 255)
        self.buf.append(b & 255)