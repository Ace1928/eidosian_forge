import math
import numpy as np
import sys
import ctypes
import warnings
from collections import namedtuple
import llvmlite.binding as ll
from llvmlite import ir
from numba import literal_unroll
from numba.core.extending import (
from numba.core import errors
from numba.core import types, utils
from numba.core.unsafe.bytes import grab_byte, grab_uint64_t
from numba.cpython.randomimpl import (const_int, get_next_int, get_next_int32,
from ctypes import (  # noqa
@register_jitable(locals={'x': _Py_uhash_t, 'p1': _Py_uhash_t, 'p2': _Py_uhash_t, 'p3': _Py_uhash_t, 'p4': _Py_uhash_t, '_PyHASH_MODULUS': _Py_uhash_t, '_PyHASH_BITS': types.int32, '_PyLong_SHIFT': types.int32})
def _long_impl(val):
    _tmp_shift = 32 - _PyLong_SHIFT
    mask_shift = ~types.uint32(0) >> _tmp_shift
    i = 64 // _PyLong_SHIFT + 1
    x = 0
    p3 = _PyHASH_BITS - _PyLong_SHIFT
    for idx in range(i - 1, -1, -1):
        p1 = x << _PyLong_SHIFT
        p2 = p1 & _PyHASH_MODULUS
        p4 = x >> p3
        x = p2 | p4
        x += types.uint32(val >> idx * _PyLong_SHIFT & mask_shift)
        if x >= _PyHASH_MODULUS:
            x -= _PyHASH_MODULUS
    return _Py_hash_t(x)