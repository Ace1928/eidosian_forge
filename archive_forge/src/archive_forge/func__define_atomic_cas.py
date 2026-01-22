from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def _define_atomic_cas(module, ordering):
    """Define a llvm function for atomic compare-and-swap.
    The generated function is a direct wrapper of the LLVM cmpxchg with the
    difference that the a int indicate success (1) or failure (0) is returned
    and the last argument is a output pointer for storing the old value.

    Note
    ----
    On failure, the generated function behaves like an atomic load.  The loaded
    value is stored to the last argument.
    """
    ftype = ir.FunctionType(ir.IntType(32), [_word_type.as_pointer(), _word_type, _word_type, _word_type.as_pointer()])
    fn_cas = ir.Function(module, ftype, name='nrt_atomic_cas')
    [ptr, cmp, repl, oldptr] = fn_cas.args
    bb = fn_cas.append_basic_block()
    builder = ir.IRBuilder(bb)
    outtup = builder.cmpxchg(ptr, cmp, repl, ordering=ordering)
    old, ok = cgutils.unpack_tuple(builder, outtup, 2)
    builder.store(old, oldptr)
    builder.ret(builder.zext(ok, ftype.return_type))
    return fn_cas