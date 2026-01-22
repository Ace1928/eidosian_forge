import array
import contextlib
import enum
import struct
@property
def AsKey(self):
    if self.IsKey:
        return str(Key(self._Indirect(), self._byte_width))
    else:
        raise self._ConvertError(Type.KEY)