import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def array_ufunc_errmsg_formatter(dummy, ufunc, method, *inputs, **kwargs):
    """ Format the error message for when __array_ufunc__ gives up. """
    args_string = ', '.join(['{!r}'.format(arg) for arg in inputs] + ['{}={!r}'.format(k, v) for k, v in kwargs.items()])
    args = inputs + kwargs.get('out', ())
    types_string = ', '.join((repr(type(arg).__name__) for arg in args))
    return 'operand type(s) all returned NotImplemented from __array_ufunc__({!r}, {!r}, {}): {}'.format(ufunc, method, args_string, types_string)