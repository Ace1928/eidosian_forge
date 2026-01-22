import contextlib
import functools
from llvmlite.ir import instructions, types, values
def _uniop_intrinsic_float(opname):

    def wrap(fn):

        @functools.wraps(fn)
        def wrapped(self, operand, name=''):
            if not isinstance(operand.type, (types.FloatType, types.DoubleType)):
                raise TypeError('expected a float type, got %s' % operand.type)
            fn = self.module.declare_intrinsic(opname, [operand.type])
            return self.call(fn, [operand], name)
        return wrapped
    return wrap