import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _chandrupatla_iv(func, args, xatol, xrtol, fatol, frtol, maxiter, callback):
    if not callable(func):
        raise ValueError('`func` must be callable.')
    if not np.iterable(args):
        args = (args,)
    tols = np.asarray([xatol if xatol is not None else 1, xrtol if xrtol is not None else 1, fatol if fatol is not None else 1, frtol if frtol is not None else 1])
    if not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0) or np.any(np.isnan(tols)) or (tols.shape != (4,)):
        raise ValueError('Tolerances must be non-negative scalars.')
    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter < 0:
        raise ValueError('`maxiter` must be a non-negative integer.')
    if callback is not None and (not callable(callback)):
        raise ValueError('`callback` must be callable.')
    return (func, args, xatol, xrtol, fatol, frtol, maxiter, callback)