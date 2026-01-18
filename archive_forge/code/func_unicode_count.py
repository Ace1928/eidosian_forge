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
@overload_method(types.UnicodeType, 'count')
def unicode_count(src, sub, start=None, end=None):
    _count_args_types_check(start)
    _count_args_types_check(end)
    if isinstance(sub, types.UnicodeType):

        def count_impl(src, sub, start=None, end=None):
            count = 0
            src_len = len(src)
            sub_len = len(sub)
            start = _normalize_slice_idx_count(start, src_len, 0)
            end = _normalize_slice_idx_count(end, src_len, src_len)
            if end - start < 0 or start > src_len:
                return 0
            src = src[start:end]
            src_len = len(src)
            start, end = (0, src_len)
            if sub_len == 0:
                return src_len + 1
            while start + sub_len <= src_len:
                if src[start:start + sub_len] == sub:
                    count += 1
                    start += sub_len
                else:
                    start += 1
            return count
        return count_impl
    error_msg = 'The substring must be a UnicodeType, not {}'
    raise TypingError(error_msg.format(type(sub)))