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
@register_jitable(locals={'acc': _Py_uhash_t, 'lane': _Py_uhash_t, '_PyHASH_XXPRIME_5': _Py_uhash_t, '_PyHASH_XXPRIME_1': _Py_uhash_t, 'tl': _Py_uhash_t})
def _tuple_hash(tup):
    tl = len(tup)
    acc = _PyHASH_XXPRIME_5
    for x in literal_unroll(tup):
        lane = hash(x)
        if lane == _Py_uhash_t(-1):
            return -1
        acc += lane * _PyHASH_XXPRIME_2
        acc = _PyHASH_XXROTATE(acc)
        acc *= _PyHASH_XXPRIME_1
    acc += tl ^ (_PyHASH_XXPRIME_5 ^ _Py_uhash_t(3527539))
    if acc == _Py_uhash_t(-1):
        return process_return(1546275796)
    return process_return(acc)