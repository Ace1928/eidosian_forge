import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
def _list_extend_list(context, builder, sig, args):
    src = ListInstance(context, builder, sig.args[1], args[1])
    dest = ListInstance(context, builder, sig.args[0], args[0])
    src_size = src.size
    dest_size = dest.size
    nitems = builder.add(src_size, dest_size)
    dest.resize(nitems)
    dest.size = nitems
    with cgutils.for_range(builder, src_size) as loop:
        value = src.getitem(loop.index)
        value = context.cast(builder, value, src.dtype, dest.dtype)
        dest.setitem(builder.add(loop.index, dest_size), value, incref=True)
    return dest