import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _bracket_root_iv(func, a, b, min, max, factor, args, maxiter):
    if not callable(func):
        raise ValueError('`func` must be callable.')
    if not np.iterable(args):
        args = (args,)
    a = np.asarray(a)[()]
    if not np.issubdtype(a.dtype, np.number) or np.iscomplex(a).any():
        raise ValueError('`a` must be numeric and real.')
    b = a + 1 if b is None else b
    min = -np.inf if min is None else min
    max = np.inf if max is None else max
    factor = 2.0 if factor is None else factor
    a, b, min, max, factor = np.broadcast_arrays(a, b, min, max, factor)
    if not np.issubdtype(b.dtype, np.number) or np.iscomplex(b).any():
        raise ValueError('`b` must be numeric and real.')
    if not np.issubdtype(min.dtype, np.number) or np.iscomplex(min).any():
        raise ValueError('`min` must be numeric and real.')
    if not np.issubdtype(max.dtype, np.number) or np.iscomplex(max).any():
        raise ValueError('`max` must be numeric and real.')
    if not np.issubdtype(factor.dtype, np.number) or np.iscomplex(factor).any():
        raise ValueError('`factor` must be numeric and real.')
    if not np.all(factor > 1):
        raise ValueError('All elements of `factor` must be greater than 1.')
    maxiter = np.asarray(maxiter)
    message = '`maxiter` must be a non-negative integer.'
    if not np.issubdtype(maxiter.dtype, np.number) or maxiter.shape != tuple() or np.iscomplex(maxiter):
        raise ValueError(message)
    maxiter_int = int(maxiter[()])
    if not maxiter == maxiter_int or maxiter < 0:
        raise ValueError(message)
    if not np.all((min <= a) & (a < b) & (b <= max)):
        raise ValueError('`min <= a < b <= max` must be True (elementwise).')
    return (func, a, b, min, max, factor, args, maxiter)