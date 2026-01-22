from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
def argsreduce(cond, *args):
    """Clean arguments to:

    1. Ensure all arguments are iterable (arrays of dimension at least one
    2. If cond != True and size > 1, ravel(args[i]) where ravel(condition) is
       True, in 1D.

    Return list of processed arguments.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._distn_infrastructure import argsreduce
    >>> rng = np.random.default_rng()
    >>> A = rng.random((4, 5))
    >>> B = 2
    >>> C = rng.random((1, 5))
    >>> cond = np.ones(A.shape)
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> A1.shape
    (4, 5)
    >>> B1.shape
    (1,)
    >>> C1.shape
    (1, 5)
    >>> cond[2,:] = 0
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> A1.shape
    (15,)
    >>> B1.shape
    (1,)
    >>> C1.shape
    (15,)

    """
    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs]
    if np.all(cond):
        *newargs, cond = np.broadcast_arrays(*newargs, cond)
        return [arg.ravel() for arg in newargs]
    s = cond.shape
    return [arg if np.size(arg) == 1 else np.extract(cond, np.broadcast_to(arg, s)) for arg in newargs]