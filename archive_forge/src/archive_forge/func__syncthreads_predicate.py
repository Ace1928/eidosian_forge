from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.errors import RequireLiteralValue
from numba.core.typing import signature
from numba.core.extending import overload_attribute
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
def _syncthreads_predicate(typingctx, predicate, fname):
    if not isinstance(predicate, types.Integer):
        return None
    sig = signature(types.i4, types.i4)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(ir.IntType(32), (ir.IntType(32),))
        sync = cgutils.get_or_insert_function(builder.module, fnty, fname)
        return builder.call(sync, args)
    return (sig, codegen)