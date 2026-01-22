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
class f_gen(rv_continuous):
    """An F continuous random variable.

    For the noncentral F distribution, see `ncf`.

    %(before_notes)s

    See Also
    --------
    ncf

    Notes
    -----
    The F distribution with :math:`df_1 > 0` and :math:`df_2 > 0` degrees of freedom is
    the distribution of the ratio of two independent chi-squared distributions with
    :math:`df_1` and :math:`df_2` degrees of freedom, after rescaling by
    :math:`df_2 / df_1`.
    
    The probability density function for `f` is:

    .. math::

        f(x, df_1, df_2) = \\frac{df_2^{df_2/2} df_1^{df_1/2} x^{df_1 / 2-1}}
                                {(df_2+df_1 x)^{(df_1+df_2)/2}
                                 B(df_1/2, df_2/2)}

    for :math:`x > 0`.

    `f` accepts shape parameters ``dfn`` and ``dfd`` for :math:`df_1`, the degrees of
    freedom of the chi-squared distribution in the numerator, and :math:`df_2`, the
    degrees of freedom of the chi-squared distribution in the denominator, respectively.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        idfn = _ShapeInfo('dfn', False, (0, np.inf), (False, False))
        idfd = _ShapeInfo('dfd', False, (0, np.inf), (False, False))
        return [idfn, idfd]

    def _rvs(self, dfn, dfd, size=None, random_state=None):
        return random_state.f(dfn, dfd, size)

    def _pdf(self, x, dfn, dfd):
        return np.exp(self._logpdf(x, dfn, dfd))

    def _logpdf(self, x, dfn, dfd):
        n = 1.0 * dfn
        m = 1.0 * dfd
        lPx = m / 2 * np.log(m) + n / 2 * np.log(n) + sc.xlogy(n / 2 - 1, x) - ((n + m) / 2 * np.log(m + n * x) + sc.betaln(n / 2, m / 2))
        return lPx

    def _cdf(self, x, dfn, dfd):
        return sc.fdtr(dfn, dfd, x)

    def _sf(self, x, dfn, dfd):
        return sc.fdtrc(dfn, dfd, x)

    def _ppf(self, q, dfn, dfd):
        return sc.fdtri(dfn, dfd, q)

    def _stats(self, dfn, dfd):
        v1, v2 = (1.0 * dfn, 1.0 * dfd)
        v2_2, v2_4, v2_6, v2_8 = (v2 - 2.0, v2 - 4.0, v2 - 6.0, v2 - 8.0)
        mu = _lazywhere(v2 > 2, (v2, v2_2), lambda v2, v2_2: v2 / v2_2, np.inf)
        mu2 = _lazywhere(v2 > 4, (v1, v2, v2_2, v2_4), lambda v1, v2, v2_2, v2_4: 2 * v2 * v2 * (v1 + v2_2) / (v1 * v2_2 ** 2 * v2_4), np.inf)
        g1 = _lazywhere(v2 > 6, (v1, v2_2, v2_4, v2_6), lambda v1, v2_2, v2_4, v2_6: (2 * v1 + v2_2) / v2_6 * np.sqrt(v2_4 / (v1 * (v1 + v2_2))), np.nan)
        g1 *= np.sqrt(8.0)
        g2 = _lazywhere(v2 > 8, (g1, v2_6, v2_8), lambda g1, v2_6, v2_8: (8 + g1 * g1 * v2_6) / v2_8, np.nan)
        g2 *= 3.0 / 2.0
        return (mu, mu2, g1, g2)

    def _entropy(self, dfn, dfd):
        half_dfn = 0.5 * dfn
        half_dfd = 0.5 * dfd
        half_sum = 0.5 * (dfn + dfd)
        return np.log(dfd) - np.log(dfn) + sc.betaln(half_dfn, half_dfd) + (1 - half_dfn) * sc.psi(half_dfn) - (1 + half_dfd) * sc.psi(half_dfd) + half_sum * sc.psi(half_sum)