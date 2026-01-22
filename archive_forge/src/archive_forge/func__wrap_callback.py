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
def _wrap_callback(callback, method=None):
    """Wrap a user-provided callback so that attributes can be attached."""
    if callback is None or method in {'tnc', 'slsqp', 'cobyla'}:
        return callback
    sig = inspect.signature(callback)
    if set(sig.parameters) == {'intermediate_result'}:

        def wrapped_callback(res):
            return callback(intermediate_result=res)
    elif method == 'trust-constr':

        def wrapped_callback(res):
            return callback(np.copy(res.x), res)
    elif method == 'differential_evolution':

        def wrapped_callback(res):
            return callback(np.copy(res.x), res.convergence)
    else:

        def wrapped_callback(res):
            return callback(np.copy(res.x))
    wrapped_callback.stop_iteration = False
    return wrapped_callback