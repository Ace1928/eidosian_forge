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
@overload(operator.add)
@overload(operator.iadd)
def unicode_concat(a, b):
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeType):

        def concat_impl(a, b):
            new_length = a._length + b._length
            new_kind = _pick_kind(a._kind, b._kind)
            new_ascii = _pick_ascii(a._is_ascii, b._is_ascii)
            result = _empty_string(new_kind, new_length, new_ascii)
            for i in range(len(a)):
                _set_code_point(result, i, _get_code_point(a, i))
            for j in range(len(b)):
                _set_code_point(result, len(a) + j, _get_code_point(b, j))
            return result
        return concat_impl
    if isinstance(a, types.UnicodeType) and isinstance(b, types.UnicodeCharSeq):

        def concat_impl(a, b):
            return a + str(b)
        return concat_impl