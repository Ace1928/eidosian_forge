import array
import contextlib
import enum
import struct
def FindOrInsert(self, data, offset):
    do = (data, offset)
    index = _BinarySearch(self._pool, do, lambda a, b: a[0] < b[0])
    if index != -1:
        _, offset = self._pool[index]
        return offset
    self._pool.insert(index, do)
    return None