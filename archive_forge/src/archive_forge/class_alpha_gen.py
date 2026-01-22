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
class alpha_gen(rv_continuous):
    """An alpha continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `alpha` ([1]_, [2]_) is:

    .. math::

        f(x, a) = \\frac{1}{x^2 \\Phi(a) \\sqrt{2\\pi}} *
                  \\exp(-\\frac{1}{2} (a-1/x)^2)

    where :math:`\\Phi` is the normal CDF, :math:`x > 0`, and :math:`a > 0`.

    `alpha` takes ``a`` as a shape parameter.

    %(after_notes)s

    References
    ----------
    .. [1] Johnson, Kotz, and Balakrishnan, "Continuous Univariate
           Distributions, Volume 1", Second Edition, John Wiley and Sons,
           p. 173 (1994).
    .. [2] Anthony A. Salvia, "Reliability applications of the Alpha
           Distribution", IEEE Transactions on Reliability, Vol. R-34,
           No. 3, pp. 251-252 (1985).

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return [_ShapeInfo('a', False, (0, np.inf), (False, False))]

    def _pdf(self, x, a):
        return 1.0 / x ** 2 / _norm_cdf(a) * _norm_pdf(a - 1.0 / x)

    def _logpdf(self, x, a):
        return -2 * np.log(x) + _norm_logpdf(a - 1.0 / x) - np.log(_norm_cdf(a))

    def _cdf(self, x, a):
        return _norm_cdf(a - 1.0 / x) / _norm_cdf(a)

    def _ppf(self, q, a):
        return 1.0 / np.asarray(a - _norm_ppf(q * _norm_cdf(a)))

    def _stats(self, a):
        return [np.inf] * 2 + [np.nan] * 2