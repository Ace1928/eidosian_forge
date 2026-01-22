from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def _build_ufunc_loop_body(load, store, context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, env):
    elems = load()
    status, retval = context.call_conv.call_function(builder, func, signature.return_type, signature.args, elems)
    with builder.if_else(status.is_ok, likely=True) as (if_ok, if_error):
        with if_ok:
            store(retval)
        with if_error:
            gil = pyapi.gil_ensure()
            context.call_conv.raise_error(builder, pyapi, status)
            pyapi.gil_release(gil)
    for off, ary in zip(offsets, arrays):
        builder.store(builder.add(builder.load(off), ary.step), off)
    builder.store(builder.add(builder.load(store_offset), out.step), store_offset)
    return status.code