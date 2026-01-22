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
def _handle_capital_sigma(data, length, idx):
    """This is a translation of the function that handles the capital sigma."""
    c = 0
    j = idx - 1
    while j >= 0:
        c = _get_code_point(data, j)
        if not _PyUnicode_IsCaseIgnorable(c):
            break
        j -= 1
    final_sigma = j >= 0 and _PyUnicode_IsCased(c)
    if final_sigma:
        j = idx + 1
        while j < length:
            c = _get_code_point(data, j)
            if not _PyUnicode_IsCaseIgnorable(c):
                break
            j += 1
        final_sigma = j == length or not _PyUnicode_IsCased(c)
    return 962 if final_sigma else 963