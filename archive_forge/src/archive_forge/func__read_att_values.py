import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _read_att_values(self):
    nc_type = self.fp.read(4)
    n = self._unpack_int()
    typecode, size = TYPEMAP[nc_type]
    count = n * size
    values = self.fp.read(int(count))
    self.fp.read(-count % 4)
    if typecode != 'c':
        values = frombuffer(values, dtype='>%s' % typecode).copy()
        if values.shape == (1,):
            values = values[0]
    else:
        values = values.rstrip(b'\x00')
    return values