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
@register_jitable(locals={'a': types.uint64, 'b': types.uint64, 'c': types.uint64, 'd': types.uint64, 's': types.uint64, 't': types.uint64})
def _HALF_ROUND(a, b, c, d, s, t):
    a += b
    c += d
    b = _ROTATE(b, s) ^ a
    d = _ROTATE(d, t) ^ c
    a = _ROTATE(a, 32)
    return (a, b, c, d)