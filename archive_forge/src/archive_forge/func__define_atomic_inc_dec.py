from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def _define_atomic_inc_dec(module, op, ordering):
    """Define a llvm function for atomic increment/decrement to the given module
    Argument ``op`` is the operation "add"/"sub".  Argument ``ordering`` is
    the memory ordering.  The generated function returns the new value.
    """
    ftype = ir.FunctionType(_word_type, [_word_type.as_pointer()])
    fn_atomic = ir.Function(module, ftype, name='nrt_atomic_{0}'.format(op))
    [ptr] = fn_atomic.args
    bb = fn_atomic.append_basic_block()
    builder = ir.IRBuilder(bb)
    ONE = ir.Constant(_word_type, 1)
    if not _disable_atomicity:
        oldval = builder.atomic_rmw(op, ptr, ONE, ordering=ordering)
        res = getattr(builder, op)(oldval, ONE)
        builder.ret(res)
    else:
        oldval = builder.load(ptr)
        newval = getattr(builder, op)(oldval, ONE)
        builder.store(newval, ptr)
        builder.ret(oldval)
    return fn_atomic