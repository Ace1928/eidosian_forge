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
@intrinsic
def _malloc_string(typingctx, kind, char_bytes, length, is_ascii):
    """make empty string with data buffer of size alloc_bytes.

    Must set length and kind values for string after it is returned
    """

    def details(context, builder, signature, args):
        [kind_val, char_bytes_val, length_val, is_ascii_val] = args
        uni_str_ctor = cgutils.create_struct_proxy(types.unicode_type)
        uni_str = uni_str_ctor(context, builder)
        nbytes_val = builder.mul(char_bytes_val, builder.add(length_val, Constant(length_val.type, 1)))
        uni_str.meminfo = context.nrt.meminfo_alloc(builder, nbytes_val)
        uni_str.kind = kind_val
        uni_str.is_ascii = is_ascii_val
        uni_str.length = length_val
        uni_str.hash = context.get_constant(_Py_hash_t, -1)
        uni_str.data = context.nrt.meminfo_data(builder, uni_str.meminfo)
        uni_str.parent = cgutils.get_null_value(uni_str.parent.type)
        return uni_str._getvalue()
    sig = types.unicode_type(types.int32, types.intp, types.intp, types.uint32)
    return (sig, details)