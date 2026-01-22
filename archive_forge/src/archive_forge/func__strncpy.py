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
def _strncpy(dst, dst_offset, src, src_offset, n):
    if src._kind == dst._kind:
        byte_width = _kind_to_byte_width(src._kind)
        src_byte_offset = byte_width * src_offset
        dst_byte_offset = byte_width * dst_offset
        nbytes = n * byte_width
        memcpy_region(dst._data, dst_byte_offset, src._data, src_byte_offset, nbytes, align=1)
    else:
        for i in range(n):
            _set_code_point(dst, dst_offset + i, _get_code_point(src, src_offset + i))