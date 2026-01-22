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
def _ppf_single(self, q, *args):
    factor = 10.0
    left, right = self._get_support(*args)
    if np.isinf(left):
        left = min(-factor, right)
        while self._ppf_to_solve(left, q, *args) > 0.0:
            left, right = (left * factor, left)
    if np.isinf(right):
        right = max(factor, left)
        while self._ppf_to_solve(right, q, *args) < 0.0:
            left, right = (right, right * factor)
    return optimize.brentq(self._ppf_to_solve, left, right, args=(q,) + args, xtol=self.xtol)