import sys
import operator
import numpy as np
from llvmlite.ir import IntType, Constant
from numba.core.cgutils import is_nonelike
from numba.core.extending import (
from numba.core.imputils import (lower_constant, lower_cast, lower_builtin,
from numba.core.datamodel import register_default, StructModel
from numba.core import types, cgutils
from numba.core.utils import PYVERSION
from numba.core.pythonapi import (
from numba._helperlib import c_helpers
from numba.cpython.hashing import _Py_hash_t
from numba.core.unsafe.bytes import memcpy_region
from numba.core.errors import TypingError
from numba.cpython.unicode_support import (_Py_TOUPPER, _Py_TOLOWER, _Py_UCS4,
from numba.cpython import slicing
@register_jitable
def _normalize_slice_idx_count(arg, slice_len, default):
    """
    Used for unicode_count

    If arg < -slice_len, returns 0 (prevents circle)

    If arg is within slice, e.g -slice_len <= arg < slice_len
    returns its real index via arg % slice_len

    If arg > slice_len, returns arg (in this case count must
    return 0 if it is start index)
    """
    if arg is None:
        return default
    if -slice_len <= arg < slice_len:
        return arg % slice_len
    return 0 if arg < 0 else arg