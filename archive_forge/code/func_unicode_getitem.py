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
@overload(operator.getitem)
def unicode_getitem(s, idx):
    if isinstance(s, types.UnicodeType):
        if isinstance(idx, types.Integer):

            def getitem_char(s, idx):
                idx = normalize_str_idx(idx, len(s))
                cp = _get_code_point(s, idx)
                kind = _codepoint_to_kind(cp)
                if kind == s._kind:
                    return _get_str_slice_view(s, idx, 1)
                else:
                    is_ascii = _codepoint_is_ascii(cp)
                    ret = _empty_string(kind, 1, is_ascii)
                    _set_code_point(ret, 0, cp)
                    return ret
            return getitem_char
        elif isinstance(idx, types.SliceType):

            def getitem_slice(s, idx):
                slice_idx = _normalize_slice(idx, len(s))
                span = _slice_span(slice_idx)
                cp = _get_code_point(s, slice_idx.start)
                kind = _codepoint_to_kind(cp)
                is_ascii = _codepoint_is_ascii(cp)
                for i in range(slice_idx.start + slice_idx.step, slice_idx.stop, slice_idx.step):
                    cp = _get_code_point(s, i)
                    is_ascii &= _codepoint_is_ascii(cp)
                    new_kind = _codepoint_to_kind(cp)
                    if kind != new_kind:
                        kind = _pick_kind(kind, new_kind)
                if slice_idx.step == 1 and kind == s._kind:
                    return _get_str_slice_view(s, slice_idx.start, span)
                else:
                    ret = _empty_string(kind, span, is_ascii)
                    cur = slice_idx.start
                    for i in range(span):
                        _set_code_point(ret, i, _get_code_point(s, cur))
                        cur += slice_idx.step
                    return ret
            return getitem_slice