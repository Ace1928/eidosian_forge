import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _write_var_data(self, name):
    var = self.variables[name]
    the_beguine = self.fp.tell()
    self.fp.seek(var._begin)
    self._pack_begin(the_beguine)
    self.fp.seek(the_beguine)
    if not var.isrec:
        self.fp.write(var.data.tobytes())
        count = var.data.size * var.data.itemsize
        self._write_var_padding(var, var._vsize - count)
    else:
        if self._recs > len(var.data):
            shape = (self._recs,) + var.data.shape[1:]
            try:
                var.data.resize(shape)
            except ValueError:
                dtype = var.data.dtype
                var.__dict__['data'] = np.resize(var.data, shape).astype(dtype)
        pos0 = pos = self.fp.tell()
        for rec in var.data:
            if not rec.shape and (rec.dtype.byteorder == '<' or (rec.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                rec = rec.byteswap()
            self.fp.write(rec.tobytes())
            count = rec.size * rec.itemsize
            self._write_var_padding(var, var._vsize - count)
            pos += self._recsize
            self.fp.seek(pos)
        self.fp.seek(pos0 + var._vsize)