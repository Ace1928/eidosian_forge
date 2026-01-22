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
@register_jitable(_nrt=False)
def _cmp_region(a, a_offset, b, b_offset, n):
    if n == 0:
        return 0
    elif a_offset + n > a._length:
        return -1
    elif b_offset + n > b._length:
        return 1
    for i in range(n):
        a_chr = _get_code_point(a, a_offset + i)
        b_chr = _get_code_point(b, b_offset + i)
        if a_chr < b_chr:
            return -1
        elif a_chr > b_chr:
            return 1
    return 0