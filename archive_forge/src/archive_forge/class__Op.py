import re
import numpy
import cupy
import cupy._core._routines_manipulation as _manipulation
from cupy._core._dtype import get_dtype, _raise_if_invalid_cast
from cupy._core import internal
class _Op:

    def __init__(self, in_types, out_types, func):
        self.func = func
        self.in_types = tuple((numpy.dtype(i) for i in in_types))
        self.out_types = tuple((numpy.dtype(o) for o in out_types))
        self.sig_str = ''.join((in_t.char for in_t in self.in_types)) + '->' + ''.join((out_t.char for out_t in self.out_types))