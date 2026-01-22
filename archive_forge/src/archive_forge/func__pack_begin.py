import warnings
import weakref
from operator import mul
from platform import python_implementation
import mmap as mm
import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce
def _pack_begin(self, begin):
    if self.version_byte == 1:
        self._pack_int(begin)
    elif self.version_byte == 2:
        self._pack_int64(begin)