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
class chi2_gen(rv_continuous):
    """A chi-squared continuous random variable.

    For the noncentral chi-square distribution, see `ncx2`.

    %(before_notes)s

    See Also
    --------
    ncx2

    Notes
    -----
    The probability density function for `chi2` is:

    .. math::

        f(x, k) = \\frac{1}{2^{k/2} \\Gamma \\left( k/2 \\right)}
                   x^{k/2-1} \\exp \\left( -x/2 \\right)

    for :math:`x > 0`  and :math:`k > 0` (degrees of freedom, denoted ``df``
    in the implementation).

    `chi2` takes ``df`` as a shape parameter.

    The chi-squared distribution is a special case of the gamma
    distribution, with gamma parameters ``a = df/2``, ``loc = 0`` and
    ``scale = 2``.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('df', False, (0, np.inf), (False, False))]

    def _rvs(self, df, size=None, random_state=None):
        return random_state.chisquare(df, size)

    def _pdf(self, x, df):
        return np.exp(self._logpdf(x, df))

    def _logpdf(self, x, df):
        return sc.xlogy(df / 2.0 - 1, x) - x / 2.0 - sc.gammaln(df / 2.0) - np.log(2) * df / 2.0

    def _cdf(self, x, df):
        return sc.chdtr(df, x)

    def _sf(self, x, df):
        return sc.chdtrc(df, x)

    def _isf(self, p, df):
        return sc.chdtri(df, p)

    def _ppf(self, p, df):
        return 2 * sc.gammaincinv(df / 2, p)

    def _stats(self, df):
        mu = df
        mu2 = 2 * df
        g1 = 2 * np.sqrt(2.0 / df)
        g2 = 12.0 / df
        return (mu, mu2, g1, g2)

    def _entropy(self, df):
        half_df = 0.5 * df

        def regular_formula(half_df):
            return half_df + np.log(2) + sc.gammaln(half_df) + (1 - half_df) * sc.psi(half_df)

        def asymptotic_formula(half_df):
            c = np.log(2) + 0.5 * (1 + np.log(2 * np.pi))
            h = 0.5 / half_df
            return h * (-2 / 3 + h * (-1 / 3 + h * (-4 / 45 + h / 7.5))) + 0.5 * np.log(half_df) + c
        return _lazywhere(half_df < 125, (half_df,), regular_formula, f2=asymptotic_formula)