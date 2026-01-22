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
def _moment_error(self, theta, x, data_moments):
    loc, scale, args = self._unpack_loc_scale(theta)
    if not self._argcheck(*args) or scale <= 0:
        return inf
    dist_moments = np.array([self.moment(i + 1, *args, loc=loc, scale=scale) for i in range(len(data_moments))])
    if np.any(np.isnan(dist_moments)):
        raise ValueError("Method of moments encountered a non-finite distribution moment and cannot continue. Consider trying method='MLE'.")
    return (((data_moments - dist_moments) / np.maximum(np.abs(data_moments), 1e-08)) ** 2).sum()