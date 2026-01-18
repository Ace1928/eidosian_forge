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
@overload_method(types.UnicodeType, 'rindex')
def unicode_rindex(s, sub, start=None, end=None):
    """Implements str.rindex()"""
    unicode_idx_check_type(start, 'start')
    unicode_idx_check_type(end, 'end')
    unicode_sub_check_type(sub, 'sub')

    def rindex_impl(s, sub, start=None, end=None):
        result = s.rfind(sub, start, end)
        if result < 0:
            raise ValueError('substring not found')
        return result
    return rindex_impl