import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _differentiate_iv(func, x, args, atol, rtol, maxiter, order, initial_step, step_factor, step_direction, callback):
    if not callable(func):
        raise ValueError('`func` must be callable.')
    x = np.asarray(x)
    dtype = x.dtype if np.issubdtype(x.dtype, np.inexact) else np.float64
    if not np.iterable(args):
        args = (args,)
    if atol is None:
        atol = np.finfo(dtype).tiny
    if rtol is None:
        rtol = np.sqrt(np.finfo(dtype).eps)
    message = 'Tolerances and step parameters must be non-negative scalars.'
    tols = np.asarray([atol, rtol, initial_step, step_factor])
    if not np.issubdtype(tols.dtype, np.number) or np.any(tols < 0) or tols.shape != (4,):
        raise ValueError(message)
    initial_step, step_factor = tols[2:].astype(dtype)
    maxiter_int = int(maxiter)
    if maxiter != maxiter_int or maxiter <= 0:
        raise ValueError('`maxiter` must be a positive integer.')
    order_int = int(order)
    if order_int != order or order <= 0:
        raise ValueError('`order` must be a positive integer.')
    step_direction = np.sign(step_direction).astype(dtype)
    x, step_direction = np.broadcast_arrays(x, step_direction)
    x, step_direction = (x[()], step_direction[()])
    if callback is not None and (not callable(callback)):
        raise ValueError('`callback` must be callable.')
    return (func, x, args, atol, rtol, maxiter_int, order_int, initial_step, step_factor, step_direction, callback)