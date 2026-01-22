import array
import contextlib
import enum
import struct
def Mutate(self, value):
    """Mutates underlying string bytes in place.

    Args:
      value: New string to replace the existing one. New string must have less
        or equal UTF-8-encoded bytes than the existing one to successfully
        mutate underlying byte buffer.

    Returns:
      Whether the value was mutated or not.
    """
    encoded = value.encode('utf-8')
    n = len(encoded)
    if n <= len(self):
        self._buf[-self._byte_width:0] = _Pack(U, n, self._byte_width)
        self._buf[0:n] = encoded
        self._buf[n:len(self)] = bytearray(len(self) - n)
        return True
    return False