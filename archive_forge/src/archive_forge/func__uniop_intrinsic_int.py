import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _uniop_intrinsic_int(opname):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, operand, name=''):
            if not isinstance(operand.type, types.IntType):
                raise TypeError('expected an integer type, got %s' % operand.type)
            fn = self.module.declare_intrinsic(opname, [operand.type])
            return self.call(fn, [operand], name)
        return wrapped
    return wrap