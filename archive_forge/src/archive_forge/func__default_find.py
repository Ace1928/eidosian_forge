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
def _default_find(data, substr, start, end):
    """Left finder."""
    m = len(substr)
    if m == 0:
        return start
    gap = mlast = m - 1
    last = _get_code_point(substr, mlast)
    zero = types.intp(0)
    mask = _bloom_add(zero, last)
    for i in range(mlast):
        ch = _get_code_point(substr, i)
        mask = _bloom_add(mask, ch)
        if ch == last:
            gap = mlast - i - 1
    i = start
    while i <= end - m:
        ch = _get_code_point(data, mlast + i)
        if ch == last:
            j = 0
            while j < mlast:
                haystack_ch = _get_code_point(data, i + j)
                needle_ch = _get_code_point(substr, j)
                if haystack_ch != needle_ch:
                    break
                j += 1
            if j == mlast:
                return i
            ch = _get_code_point(data, mlast + i + 1)
            if _bloom_check(mask, ch) == 0:
                i += m
            else:
                i += gap
        else:
            ch = _get_code_point(data, mlast + i + 1)
            if _bloom_check(mask, ch) == 0:
                i += m
        i += 1
    return -1