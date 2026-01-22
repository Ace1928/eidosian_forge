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
def _reduce_func(self, args, kwds, data=None):
    """
        Return the (possibly reduced) function to optimize in order to find MLE
        estimates for the .fit method.
        """
    shapes = []
    if self.shapes:
        shapes = self.shapes.replace(',', ' ').split()
        for j, s in enumerate(shapes):
            key = 'f' + str(j)
            names = [key, 'f' + s, 'fix_' + s]
            val = _get_fixed_fit_value(kwds, names)
            if val is not None:
                kwds[key] = val
    args = list(args)
    Nargs = len(args)
    fixedn = []
    names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
    x0 = []
    for n, key in enumerate(names):
        if key in kwds:
            fixedn.append(n)
            args[n] = kwds.pop(key)
        else:
            x0.append(args[n])
    methods = {'mle', 'mm'}
    method = kwds.pop('method', 'mle').lower()
    if method == 'mm':
        n_params = len(shapes) + 2 - len(fixedn)
        exponents = np.arange(1, n_params + 1)[:, np.newaxis]
        data_moments = np.sum(data[None, :] ** exponents / len(data), axis=1)

        def objective(theta, x):
            return self._moment_error(theta, x, data_moments)
    elif method == 'mle':
        objective = self._penalized_nnlf
    else:
        raise ValueError("Method '{}' not available; must be one of {}".format(method, methods))
    if len(fixedn) == 0:
        func = objective
        restore = None
    else:
        if len(fixedn) == Nargs:
            raise ValueError('All parameters fixed. There is nothing to optimize.')

        def restore(args, theta):
            i = 0
            for n in range(Nargs):
                if n not in fixedn:
                    args[n] = theta[i]
                    i += 1
            return args

        def func(theta, x):
            newtheta = restore(args[:], theta)
            return objective(newtheta, x)
    return (x0, func, restore, args)