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
def unicode_strip_right_bound(string, chars):
    str_len = len(string)
    i = 0
    if chars is not None:
        for i in range(str_len - 1, -1, -1):
            if string[i] not in chars:
                i += 1
                break
    else:
        for i in range(str_len - 1, -1, -1):
            if not _PyUnicode_IsSpace(string[i]):
                i += 1
                break
    return i