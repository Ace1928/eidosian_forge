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
@overload_method(types.UnicodeType, 'endswith')
def unicode_endswith(s, substr, start=None, end=None):
    if not (start is None or isinstance(start, (types.Omitted, types.Integer, types.NoneType))):
        raise TypingError('The arg must be a Integer or None')
    if not (end is None or isinstance(end, (types.Omitted, types.Integer, types.NoneType))):
        raise TypingError('The arg must be a Integer or None')
    if isinstance(substr, (types.Tuple, types.UniTuple)):

        def endswith_impl(s, substr, start=None, end=None):
            for item in substr:
                if s.endswith(item, start, end) is True:
                    return True
            return False
        return endswith_impl
    if isinstance(substr, types.UnicodeType):

        def endswith_impl(s, substr, start=None, end=None):
            length = len(s)
            sub_length = len(substr)
            if start is None:
                start = 0
            if end is None:
                end = length
            start, end = _adjust_indices(length, start, end)
            if end - start < sub_length:
                return False
            if sub_length == 0:
                return True
            s = s[start:end]
            offset = len(s) - sub_length
            return _cmp_region(s, offset, substr, 0, sub_length) == 0
        return endswith_impl
    if isinstance(substr, types.UnicodeCharSeq):

        def endswith_impl(s, substr, start=None, end=None):
            return s.endswith(str(substr), start, end)
        return endswith_impl