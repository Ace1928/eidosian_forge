import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _read_var(self):
    name = self._unpack_string().decode('latin1')
    dimensions = []
    shape = []
    dims = self._unpack_int()
    for i in range(dims):
        dimid = self._unpack_int()
        dimname = self._dims[dimid]
        dimensions.append(dimname)
        dim = self.dimensions[dimname]
        shape.append(dim)
    dimensions = tuple(dimensions)
    shape = tuple(shape)
    attributes = self._read_att_array()
    nc_type = self.fp.read(4)
    vsize = self._unpack_int()
    begin = [self._unpack_int, self._unpack_int64][self.version_byte - 1]()
    typecode, size = TYPEMAP[nc_type]
    dtype_ = '>%s' % typecode
    return (name, dimensions, shape, attributes, typecode, size, dtype_, begin, vsize)