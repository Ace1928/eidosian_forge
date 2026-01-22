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
class maxwell_gen(rv_continuous):
    """A Maxwell continuous random variable.

    %(before_notes)s

    Notes
    -----
    A special case of a `chi` distribution,  with ``df=3``, ``loc=0.0``,
    and given ``scale = a``, where ``a`` is the parameter used in the
    Mathworld description [1]_.

    The probability density function for `maxwell` is:

    .. math::

        f(x) = \\sqrt{2/\\pi}x^2 \\exp(-x^2/2)

    for :math:`x >= 0`.

    %(after_notes)s

    References
    ----------
    .. [1] http://mathworld.wolfram.com/MaxwellDistribution.html

    %(example)s
    """

    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return chi.rvs(3.0, size=size, random_state=random_state)

    def _pdf(self, x):
        return _SQRT_2_OVER_PI * x * x * np.exp(-x * x / 2.0)

    def _logpdf(self, x):
        with np.errstate(divide='ignore'):
            return _LOG_SQRT_2_OVER_PI + 2 * np.log(x) - 0.5 * x * x

    def _cdf(self, x):
        return sc.gammainc(1.5, x * x / 2.0)

    def _ppf(self, q):
        return np.sqrt(2 * sc.gammaincinv(1.5, q))

    def _sf(self, x):
        return sc.gammaincc(1.5, x * x / 2.0)

    def _isf(self, q):
        return np.sqrt(2 * sc.gammainccinv(1.5, q))

    def _stats(self):
        val = 3 * np.pi - 8
        return (2 * np.sqrt(2.0 / np.pi), 3 - 8 / np.pi, np.sqrt(2) * (32 - 10 * np.pi) / val ** 1.5, (-12 * np.pi * np.pi + 160 * np.pi - 384) / val ** 2.0)

    def _entropy(self):
        return _EULER + 0.5 * np.log(2 * np.pi) - 0.5