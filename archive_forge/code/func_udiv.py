import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop('udiv')
def udiv(self, lhs, rhs, name=''):
    """
        Unsigned integer division:
            name = lhs / rhs
        """