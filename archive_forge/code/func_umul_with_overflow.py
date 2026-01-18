import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop_with_overflow('umul')
def umul_with_overflow(self, lhs, rhs, name=''):
    """
        Unsigned integer multiplication with overflow:
            name = {result, overflow bit} = lhs * rhs
        """