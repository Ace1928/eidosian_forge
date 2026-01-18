import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_castop('zext')
def zext(self, value, typ, name=''):
    """
        Zero-extending integer upcast to a larger type:
            name = (typ) value
        """