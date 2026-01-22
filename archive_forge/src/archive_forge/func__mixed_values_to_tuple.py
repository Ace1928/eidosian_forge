import ctypes
import operator
from enum import IntEnum
from llvmlite import ir
from numba import _helperlib
from numba.core.extending import (
from numba.core.imputils import iternext_impl, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.types import (
from numba.core.imputils import impl_ret_borrowed, RefType
from numba.core.errors import TypingError, LoweringError
from numba.core import typing
from numba.typed.typedobjectutils import (_as_bytes, _cast, _nonoptional,
@intrinsic
def _mixed_values_to_tuple(tyctx, d):
    keys = [x for x in d.literal_value.keys()]
    literal_tys = [x for x in d.literal_value.values()]

    def impl(cgctx, builder, sig, args):
        lld, = args
        impl = cgctx.get_function('static_getitem', types.none(d, types.literal('dummy')))
        items = []
        for k in range(len(keys)):
            item = impl(builder, (lld, k))
            casted = cgctx.cast(builder, item, literal_tys[k], d.types[k])
            items.append(casted)
            cgctx.nrt.incref(builder, d.types[k], item)
        ret = cgctx.make_tuple(builder, sig.return_type, items)
        return ret
    sig = types.Tuple(d.types)(d)
    return (sig, impl)