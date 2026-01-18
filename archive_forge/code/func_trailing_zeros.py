from numba.core import types, errors
from numba.core.extending import intrinsic
from llvmlite import ir
@intrinsic
def trailing_zeros(typeingctx, src):
    """Counts trailing zeros in the binary representation of an integer."""
    if not isinstance(src, types.Integer):
        msg = f"trailing_zeros is only defined for integers, but value passed was '{src}'."
        raise errors.NumbaTypeError(msg)

    def codegen(context, builder, signature, args):
        [src] = args
        return builder.cttz(src, ir.Constant(ir.IntType(1), 0))
    return (src(src), codegen)