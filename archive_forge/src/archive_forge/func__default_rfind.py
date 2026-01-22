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
def _default_rfind(data, substr, start, end):
    """Right finder."""
    m = len(substr)
    if m == 0:
        return end
    skip = mlast = m - 1
    mfirst = _get_code_point(substr, 0)
    mask = _bloom_add(0, mfirst)
    i = mlast
    while i > 0:
        ch = _get_code_point(substr, i)
        mask = _bloom_add(mask, ch)
        if ch == mfirst:
            skip = i - 1
        i -= 1
    i = end - m
    while i >= start:
        ch = _get_code_point(data, i)
        if ch == mfirst:
            j = mlast
            while j > 0:
                haystack_ch = _get_code_point(data, i + j)
                needle_ch = _get_code_point(substr, j)
                if haystack_ch != needle_ch:
                    break
                j -= 1
            if j == 0:
                return i
            ch = _get_code_point(data, i - 1)
            if i > start and _bloom_check(mask, ch) == 0:
                i -= m
            else:
                i -= skip
        else:
            ch = _get_code_point(data, i - 1)
            if i > start and _bloom_check(mask, ch) == 0:
                i -= m
        i -= 1
    return -1