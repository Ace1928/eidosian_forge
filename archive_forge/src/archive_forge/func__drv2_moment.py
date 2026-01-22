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
def _drv2_moment(self, n, *args):
    """Non-central moment of discrete distribution."""

    def fun(x):
        return np.power(x, n) * self._pmf(x, *args)
    _a, _b = self._get_support(*args)
    return _expect(fun, _a, _b, self.ppf(0.5, *args), self.inc)