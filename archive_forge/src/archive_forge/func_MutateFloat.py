import array
import contextlib
import enum
import struct
def MutateFloat(self, value):
    """Mutates underlying floating point value bytes in place.

    Args:
      value: New float value. It must fit to the byte size of the existing
        encoded value.

    Returns:
      Whether the value was mutated or not.
    """
    if self._type is Type.FLOAT:
        return _Mutate(F, self._buf, value, self._parent_width, BitWidth.B(self._parent_width))
    elif self._type is Type.INDIRECT_FLOAT:
        return _Mutate(F, self._Indirect(), value, self._byte_width, BitWidth.B(self._byte_width))
    else:
        return False