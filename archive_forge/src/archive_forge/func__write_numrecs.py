import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _write_numrecs(self):
    for var in self.variables.values():
        if var.isrec and len(var.data) > self._recs:
            self.__dict__['_recs'] = len(var.data)
    self._pack_int(self._recs)