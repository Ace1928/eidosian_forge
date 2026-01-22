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
def _call_dict_free(context, builder, ptr):
    """Call numba_dict_free(ptr)
    """
    fnty = ir.FunctionType(ir.VoidType(), [ll_dict_type])
    free = cgutils.get_or_insert_function(builder.module, fnty, 'numba_dict_free')
    builder.call(free, [ptr])