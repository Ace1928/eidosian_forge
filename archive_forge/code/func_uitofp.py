import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('uitofp')
def uitofp(self, value, typ, name=''):
    """
        Convert unsigned integer to floating-point:
            name = (typ) value
        """