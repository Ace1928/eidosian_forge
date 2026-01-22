import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def _call_callback_maybe_halt(callback, res):
    """Call wrapped callback; return True if minimization should stop.

    Parameters
    ----------
    callback : callable or None
        A user-provided callback wrapped with `_wrap_callback`
    res : OptimizeResult
        Information about the current iterate

    Returns
    -------
    halt : bool
        True if minimization should stop

    """
    if callback is None:
        return False
    try:
        callback(res)
        return False
    except StopIteration:
        callback.stop_iteration = True
        return True