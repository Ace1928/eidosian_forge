import array
import contextlib
import enum
import struct
@property
def AsMap(self):
    if self.IsMap:
        return Map(self._Indirect(), self._byte_width)
    else:
        raise self._ConvertError(Type.MAP)