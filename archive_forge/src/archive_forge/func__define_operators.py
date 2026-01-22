import numbers
from functools import reduce
from operator import mul
import numpy as np
def _define_operators(cls):
    """Decorator which adds support for some Python operators."""

    def _wrap(cls, op, inplace=False, unary=False):

        def fn_unary_op(self):
            try:
                return self._op(op)
            except SystemError as e:
                message = "Numpy returned an uninformative error. It possibly should be 'Integers to negative integer powers are not allowed.' See https://github.com/numpy/numpy/issues/19634 for details."
                raise ValueError(message) from e

        def fn_binary_op(self, value):
            try:
                return self._op(op, value, inplace=inplace)
            except SystemError as e:
                message = "Numpy returned an uninformative error. It possibly should be 'Integers to negative integer powers are not allowed.' See https://github.com/numpy/numpy/issues/19634 for details."
                raise ValueError(message) from e
        setattr(cls, op, fn_unary_op if unary else fn_binary_op)
        fn = getattr(cls, op)
        fn.__name__ = op
        fn.__doc__ = getattr(np.ndarray, op).__doc__
    for op in ('__add__', '__sub__', '__mul__', '__mod__', '__pow__', '__floordiv__', '__truediv__', '__lshift__', '__rshift__', '__or__', '__and__', '__xor__'):
        _wrap(cls, op=op, inplace=False)
        _wrap(cls, op=f'__i{op.strip('_')}__', inplace=True)
    for op in ('__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__'):
        _wrap(cls, op)
    for op in ('__neg__', '__abs__', '__invert__'):
        _wrap(cls, op, unary=True)
    return cls