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
def _as_meminfo(typingctx, dctobj):
    """Returns the MemInfoPointer of a dictionary.
    """
    if not isinstance(dctobj, types.DictType):
        raise TypingError('expected *dctobj* to be a DictType')

    def codegen(context, builder, sig, args):
        [td] = sig.args
        [d] = args
        context.nrt.incref(builder, td, d)
        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder, value=d)
        return dstruct.meminfo
    sig = _meminfo_dictptr(dctobj)
    return (sig, codegen)