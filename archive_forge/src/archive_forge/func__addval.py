import ast as _ast
import io as _io
import os as _os
import collections.abc
def _addval(self, val):
    with _io.open(self._datfile, 'rb+') as f:
        f.seek(0, 2)
        pos = int(f.tell())
        npos = (pos + _BLOCKSIZE - 1) // _BLOCKSIZE * _BLOCKSIZE
        f.write(b'\x00' * (npos - pos))
        pos = npos
        f.write(val)
    return (pos, len(val))