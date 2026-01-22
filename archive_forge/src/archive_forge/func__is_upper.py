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
def _is_upper(is_lower, is_upper, is_title):

    def impl(a):
        l = len(a)
        if l == 1:
            return is_upper(_get_code_point(a, 0))
        if l == 0:
            return False
        cased = False
        for idx in range(l):
            code_point = _get_code_point(a, idx)
            if is_lower(code_point) or is_title(code_point):
                return False
            elif not cased and is_upper(code_point):
                cased = True
        return cased
    return impl