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
class burr12_gen(rv_continuous):
    """A Burr (Type XII) continuous random variable.

    %(before_notes)s

    See Also
    --------
    fisk : a special case of either `burr` or `burr12` with ``d=1``
    burr : Burr Type III distribution

    Notes
    -----
    The probability density function for `burr12` is:

    .. math::

        f(x; c, d) = c d \\frac{x^{c-1}}
                              {(1 + x^c)^{d + 1}}

    for :math:`x >= 0` and :math:`c, d > 0`.

    `burr12` takes ``c`` and ``d`` as shape parameters for :math:`c`
    and :math:`d`.

    This is the PDF corresponding to the twelfth CDF given in Burr's list;
    specifically, it is equation (20) in Burr's paper [1]_.

    %(after_notes)s

    The Burr type 12 distribution is also sometimes referred to as
    the Singh-Maddala distribution from NIST [2]_.

    References
    ----------
    .. [1] Burr, I. W. "Cumulative frequency functions", Annals of
       Mathematical Statistics, 13(2), pp 215-232 (1942).

    .. [2] https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/b12pdf.htm

    .. [3] "Burr distribution",
       https://en.wikipedia.org/wiki/Burr_distribution

    %(example)s

    """

    def _shape_info(self):
        ic = _ShapeInfo('c', False, (0, np.inf), (False, False))
        id = _ShapeInfo('d', False, (0, np.inf), (False, False))
        return [ic, id]

    def _pdf(self, x, c, d):
        return np.exp(self._logpdf(x, c, d))

    def _logpdf(self, x, c, d):
        return np.log(c) + np.log(d) + sc.xlogy(c - 1, x) + sc.xlog1py(-d - 1, x ** c)

    def _cdf(self, x, c, d):
        return -sc.expm1(self._logsf(x, c, d))

    def _logcdf(self, x, c, d):
        return sc.log1p(-(1 + x ** c) ** (-d))

    def _sf(self, x, c, d):
        return np.exp(self._logsf(x, c, d))

    def _logsf(self, x, c, d):
        return sc.xlog1py(-d, x ** c)

    def _ppf(self, q, c, d):
        return sc.expm1(-1 / d * sc.log1p(-q)) ** (1 / c)

    def _munp(self, n, c, d):

        def moment_if_exists(n, c, d):
            nc = 1.0 * n / c
            return d * sc.beta(1.0 + nc, d - nc)
        return _lazywhere(c * d > n, (n, c, d), moment_if_exists, fillvalue=np.nan)