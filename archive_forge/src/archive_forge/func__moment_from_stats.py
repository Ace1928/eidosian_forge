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
def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    if n == 0:
        return 1.0
    elif n == 1:
        if mu is None:
            val = moment_func(1, *args)
        else:
            val = mu
    elif n == 2:
        if mu2 is None or mu is None:
            val = moment_func(2, *args)
        else:
            val = mu2 + mu * mu
    elif n == 3:
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)
        else:
            mu3 = g1 * np.power(mu2, 1.5)
            val = mu3 + 3 * mu * mu2 + mu * mu * mu
    elif n == 4:
        if g1 is None or g2 is None or mu2 is None or (mu is None):
            val = moment_func(4, *args)
        else:
            mu4 = (g2 + 3.0) * mu2 ** 2.0
            mu3 = g1 * np.power(mu2, 1.5)
            val = mu4 + 4 * mu * mu3 + 6 * mu * mu * mu2 + mu * mu * mu * mu
    else:
        val = moment_func(n, *args)
    return val