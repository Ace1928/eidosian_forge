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
def _rsplit_char(data, ch, maxsplit):
    result = []
    ch_code_point = _get_code_point(ch, 0)
    i = j = len(data) - 1
    while i >= 0 and maxsplit > 0:
        data_code_point = _get_code_point(data, i)
        if data_code_point == ch_code_point:
            result.append(data[i + 1:j + 1])
            j = i = i - 1
            maxsplit -= 1
        i -= 1
    if j >= -1:
        result.append(data[0:j + 1])
    return result[::-1]