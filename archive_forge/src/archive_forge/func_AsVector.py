import array
import contextlib
import enum
import struct
@property
def AsVector(self):
    if self.IsVector:
        return Vector(self._Indirect(), self._byte_width)
    else:
        raise self._ConvertError(Type.VECTOR)