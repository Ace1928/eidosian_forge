import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _uniop_intrinsic_with_flag(opname):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, operand, flag, name=''):
            if not isinstance(operand.type, types.IntType):
                raise TypeError('expected an integer type, got %s' % operand.type)
            if not (isinstance(flag.type, types.IntType) and flag.type.width == 1):
                raise TypeError('expected an i1 type, got %s' % flag.type)
            fn = self.module.declare_intrinsic(opname, [operand.type, flag.type])
            return self.call(fn, [operand, flag], name)
        return wrapped
    return wrap