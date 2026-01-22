import array
import contextlib
import enum
import struct
def _EndVector(self, start, typed, fixed):
    """Finishes vector construction by encodung its elements."""
    vec = self._CreateVector(self._stack[start:], typed, fixed)
    del self._stack[start:]
    self._stack.append(vec)
    return vec.Value