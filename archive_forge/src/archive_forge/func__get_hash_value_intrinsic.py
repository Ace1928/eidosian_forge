import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
@intrinsic
def _get_hash_value_intrinsic(typingctx, value):

    def impl(context, builder, typ, args):
        return get_hash_value(context, builder, value, args[0])
    fnty = typingctx.resolve_value_type(hash)
    sig = fnty.get_call_type(typingctx, (value,), {})
    return (sig, impl)