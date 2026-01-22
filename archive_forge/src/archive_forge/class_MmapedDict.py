import json
import mmap
import os
import struct
from typing import List
class MmapedDict:
    """A dict of doubles, backed by an mmapped file.

    The file starts with a 4 byte int, indicating how much of it is used.
    Then 4 bytes of padding.
    There's then a number of entries, consisting of a 4 byte int which is the
    size of the next field, a utf-8 encoded string key, padding to a 8 byte
    alignment, and then a 8 byte float which is the value and a 8 byte float
    which is a UNIX timestamp in seconds.

    Not thread safe.
    """

    def __init__(self, filename, read_mode=False):
        self._f = open(filename, 'rb' if read_mode else 'a+b')
        self._fname = filename
        capacity = os.fstat(self._f.fileno()).st_size
        if capacity == 0:
            self._f.truncate(_INITIAL_MMAP_SIZE)
            capacity = _INITIAL_MMAP_SIZE
        self._capacity = capacity
        self._m = mmap.mmap(self._f.fileno(), self._capacity, access=mmap.ACCESS_READ if read_mode else mmap.ACCESS_WRITE)
        self._positions = {}
        self._used = _unpack_integer(self._m, 0)[0]
        if self._used == 0:
            self._used = 8
            _pack_integer(self._m, 0, self._used)
        elif not read_mode:
            for key, _, _, pos in self._read_all_values():
                self._positions[key] = pos

    @staticmethod
    def read_all_values_from_file(filename):
        with open(filename, 'rb') as infp:
            data = infp.read(mmap.PAGESIZE)
            used = _unpack_integer(data, 0)[0]
            if used > len(data):
                data += infp.read(used - len(data))
        return _read_all_values(data, used)

    def _init_value(self, key):
        """Initialize a value. Lock must be held by caller."""
        encoded = key.encode('utf-8')
        padded = encoded + b' ' * (8 - (len(encoded) + 4) % 8)
        value = struct.pack(f'i{len(padded)}sdd'.encode(), len(encoded), padded, 0.0, 0.0)
        while self._used + len(value) > self._capacity:
            self._capacity *= 2
            self._f.truncate(self._capacity)
            self._m = mmap.mmap(self._f.fileno(), self._capacity)
        self._m[self._used:self._used + len(value)] = value
        self._used += len(value)
        _pack_integer(self._m, 0, self._used)
        self._positions[key] = self._used - 16

    def _read_all_values(self):
        """Yield (key, value, pos). No locking is performed."""
        return _read_all_values(data=self._m, used=self._used)

    def read_all_values(self):
        """Yield (key, value, timestamp). No locking is performed."""
        for k, v, ts, _ in self._read_all_values():
            yield (k, v, ts)

    def read_value(self, key):
        if key not in self._positions:
            self._init_value(key)
        pos = self._positions[key]
        return _unpack_two_doubles(self._m, pos)

    def write_value(self, key, value, timestamp):
        if key not in self._positions:
            self._init_value(key)
        pos = self._positions[key]
        _pack_two_doubles(self._m, pos, value, timestamp)

    def close(self):
        if self._f:
            self._m.close()
            self._m = None
            self._f.close()
            self._f = None