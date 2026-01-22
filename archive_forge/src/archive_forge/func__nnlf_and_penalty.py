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
def _nnlf_and_penalty(self, x, args):
    """
        Compute the penalized negative log-likelihood for the
        "standardized" data (i.e. already shifted by loc and
        scaled by scale) for the shape parameters in `args`.

        `x` can be a 1D numpy array or a CensoredData instance.
        """
    if isinstance(x, CensoredData):
        xs = x._supported(*self._get_support(*args))
        n_bad = len(x) - len(xs)
        i1, i2 = xs._interval.T
        terms = [self._logpdf(xs._uncensored, *args), self._logcdf(xs._left, *args), self._logsf(xs._right, *args), np.log(self._delta_cdf(i1, i2, *args))]
    else:
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        terms = [self._logpdf(x, *args)]
    totals, bad_counts = zip(*[_sum_finite(term) for term in terms])
    total = sum(totals)
    n_bad += sum(bad_counts)
    return -total + n_bad * _LOGXMAX * 100