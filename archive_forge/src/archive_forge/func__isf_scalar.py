import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
def _isf_scalar(q, a, b):

    def eq(x, a, b, q):
        return self._sf(x, a, b) - q
    xm = self.mean(a, b)
    em = eq(xm, a, b, q)
    if em == 0:
        return xm
    if em > 0:
        delta = 1
        left = xm
        right = xm + delta
        while eq(right, a, b, q) > 0:
            delta = 2 * delta
            right = xm + delta
    else:
        delta = 1
        right = xm
        left = xm - delta
        while eq(left, a, b, q) < 0:
            delta = 2 * delta
            left = xm - delta
    result = optimize.brentq(eq, left, right, args=(a, b, q), xtol=self.xtol)
    return result