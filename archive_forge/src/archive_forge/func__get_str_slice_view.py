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
def _get_str_slice_view(typingctx, src_t, start_t, length_t):
    """Create a slice of a unicode string using a view of its data to avoid
    extra allocation.
    """
    assert src_t == types.unicode_type

    def codegen(context, builder, sig, args):
        src, start, length = args
        in_str = cgutils.create_struct_proxy(types.unicode_type)(context, builder, value=src)
        view_str = cgutils.create_struct_proxy(types.unicode_type)(context, builder)
        view_str.meminfo = in_str.meminfo
        view_str.kind = in_str.kind
        view_str.is_ascii = in_str.is_ascii
        view_str.length = length
        view_str.hash = context.get_constant(_Py_hash_t, -1)
        bw_typ = context.typing_context.resolve_value_type(_kind_to_byte_width)
        bw_sig = bw_typ.get_call_type(context.typing_context, (types.int32,), {})
        bw_impl = context.get_function(bw_typ, bw_sig)
        byte_width = bw_impl(builder, (in_str.kind,))
        offset = builder.mul(start, byte_width)
        view_str.data = builder.gep(in_str.data, [offset])
        view_str.parent = cgutils.get_null_value(view_str.parent.type)
        if context.enable_nrt:
            context.nrt.incref(builder, sig.args[0], src)
        return view_str._getvalue()
    sig = types.unicode_type(types.unicode_type, types.intp, types.intp)
    return (sig, codegen)