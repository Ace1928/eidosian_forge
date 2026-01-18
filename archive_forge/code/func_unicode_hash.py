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
@overload_method(types.UnicodeType, '__hash__')
def unicode_hash(val):
    from numba.cpython.unicode import _kind_to_byte_width

    def impl(val):
        kindwidth = _kind_to_byte_width(val._kind)
        _len = len(val)
        current_hash = val._hash
        if current_hash != -1:
            return current_hash
        else:
            return _Py_HashBytes(val._data, kindwidth * _len)
    return impl