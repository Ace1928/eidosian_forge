import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _binop_with_overflow(opname, cls=instructions.Instruction):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, lhs, rhs, name=''):
            if lhs.type != rhs.type:
                raise ValueError('Operands must be the same type, got (%s, %s)' % (lhs.type, rhs.type))
            ty = lhs.type
            if not isinstance(ty, types.IntType):
                raise TypeError('expected an integer type, got %s' % (ty,))
            bool_ty = types.IntType(1)
            mod = self.module
            fnty = types.FunctionType(types.LiteralStructType([ty, bool_ty]), [ty, ty])
            fn = mod.declare_intrinsic('llvm.%s.with.overflow' % (opname,), [ty], fnty)
            ret = self.call(fn, [lhs, rhs], name=name)
            return ret
        return wrapped
    return wrap