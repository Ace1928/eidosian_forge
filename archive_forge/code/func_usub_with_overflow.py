import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop_with_overflow('usub')
def usub_with_overflow(self, lhs, rhs, name=''):
    """
        Unsigned integer subtraction with overflow:
            name = {result, overflow bit} = lhs - rhs
        """