import array
import contextlib
import enum
import struct
@property
def AsStringBytes(self):
    if self.IsString:
        return String(self._Indirect(), self._byte_width).Bytes
    elif self.IsKey:
        return self.AsKeyBytes
    else:
        raise self._ConvertError(Type.STRING)