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
def _penalized_nlpsf(self, theta, x):
    """Penalized negative log product spacing function.
        i.e., - sum (log (diff (cdf (x, theta))), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        Follows reference [1] of scipy.stats.fit
        """
    loc, scale, args = self._unpack_loc_scale(theta)
    if not self._argcheck(*args) or scale <= 0:
        return inf
    x = (np.sort(x) - loc) / scale

    def log_psf(x, *args):
        x, lj = np.unique(x, return_counts=True)
        cdf_data = self._cdf(x, *args) if x.size else []
        if not (x.size and 1 - cdf_data[-1] <= 0):
            cdf = np.concatenate(([0], cdf_data, [1]))
            lj = np.concatenate((lj, [1]))
        else:
            cdf = np.concatenate(([0], cdf_data))
        return lj * np.log(np.diff(cdf) / lj)
    return self._nlff_and_penalty(x, args, log_psf)