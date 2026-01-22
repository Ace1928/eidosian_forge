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
def _set_copy(typingctx, s):
    sig = s(s)

    def set_copy(context, builder, sig, args):
        inst = SetInstance(context, builder, sig.args[0], args[0])
        other = inst.copy()
        return impl_ret_new_ref(context, builder, sig.return_type, other.value)
    return (sig, set_copy)