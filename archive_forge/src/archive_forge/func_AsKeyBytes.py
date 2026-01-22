import array
import contextlib
import enum
import struct
@property
def AsKeyBytes(self):
    if self.IsKey:
        return Key(self._Indirect(), self._byte_width).Bytes
    else:
        raise self._ConvertError(Type.KEY)