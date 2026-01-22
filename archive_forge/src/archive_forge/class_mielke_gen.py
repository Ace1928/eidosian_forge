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
class mielke_gen(rv_continuous):
    """A Mielke Beta-Kappa / Dagum continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `mielke` is:

    .. math::

        f(x, k, s) = \\frac{k x^{k-1}}{(1+x^s)^{1+k/s}}

    for :math:`x > 0` and :math:`k, s > 0`. The distribution is sometimes
    called Dagum distribution ([2]_). It was already defined in [3]_, called
    a Burr Type III distribution (`burr` with parameters ``c=s`` and
    ``d=k/s``).

    `mielke` takes ``k`` and ``s`` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] Mielke, P.W., 1973 "Another Family of Distributions for Describing
           and Analyzing Precipitation Data." J. Appl. Meteor., 12, 275-280
    .. [2] Dagum, C., 1977 "A new model for personal income distribution."
           Economie Appliquee, 33, 327-367.
    .. [3] Burr, I. W. "Cumulative frequency functions", Annals of
           Mathematical Statistics, 13(2), pp 215-232 (1942).

    %(example)s

    """

    def _shape_info(self):
        ik = _ShapeInfo('k', False, (0, np.inf), (False, False))
        i_s = _ShapeInfo('s', False, (0, np.inf), (False, False))
        return [ik, i_s]

    def _pdf(self, x, k, s):
        return k * x ** (k - 1.0) / (1.0 + x ** s) ** (1.0 + k * 1.0 / s)

    def _logpdf(self, x, k, s):
        with np.errstate(divide='ignore'):
            return np.log(k) + np.log(x) * (k - 1) - np.log1p(x ** s) * (1 + k / s)

    def _cdf(self, x, k, s):
        return x ** k / (1.0 + x ** s) ** (k * 1.0 / s)

    def _ppf(self, q, k, s):
        qsk = pow(q, s * 1.0 / k)
        return pow(qsk / (1.0 - qsk), 1.0 / s)

    def _munp(self, n, k, s):

        def nth_moment(n, k, s):
            return sc.gamma((k + n) / s) * sc.gamma(1 - n / s) / sc.gamma(k / s)
        return _lazywhere(n < s, (n, k, s), nth_moment, np.inf)