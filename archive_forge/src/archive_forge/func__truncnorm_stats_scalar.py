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
def _truncnorm_stats_scalar(a, b, pA, pB, moments):
    m1 = pA - pB
    mu = m1
    probs = [pA, -pB]
    vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x * y, fillvalue=0)
    m2 = 1 + np.sum(vals)
    vals = _lazywhere(probs, [probs, [a - mu, b - mu]], lambda x, y: x * y, fillvalue=0)
    mu2 = 1 + np.sum(vals)
    vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x * y ** 2, fillvalue=0)
    m3 = 2 * m1 + np.sum(vals)
    vals = _lazywhere(probs, [probs, [a, b]], lambda x, y: x * y ** 3, fillvalue=0)
    m4 = 3 * m2 + np.sum(vals)
    mu3 = m3 + m1 * (-3 * m2 + 2 * m1 ** 2)
    g1 = mu3 / np.power(mu2, 1.5)
    mu4 = m4 + m1 * (-4 * m3 + 3 * m1 * (2 * m2 - m1 ** 2))
    g2 = mu4 / mu2 ** 2 - 3
    return (mu, mu2, g1, g2)