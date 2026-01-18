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
@overload_method(types.UnicodeType, 'lstrip')
def unicode_lstrip(string, chars=None):
    if isinstance(chars, types.UnicodeCharSeq):

        def lstrip_impl(string, chars=None):
            return string.lstrip(str(chars))
        return lstrip_impl
    unicode_strip_types_check(chars)

    def lstrip_impl(string, chars=None):
        return string[unicode_strip_left_bound(string, chars):]
    return lstrip_impl