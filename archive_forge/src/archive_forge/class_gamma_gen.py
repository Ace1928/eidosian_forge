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
class gamma_gen(rv_continuous):
    """A gamma continuous random variable.

    %(before_notes)s

    See Also
    --------
    erlang, expon

    Notes
    -----
    The probability density function for `gamma` is:

    .. math::

        f(x, a) = \\frac{x^{a-1} e^{-x}}{\\Gamma(a)}

    for :math:`x \\ge 0`, :math:`a > 0`. Here :math:`\\Gamma(a)` refers to the
    gamma function.

    `gamma` takes ``a`` as a shape parameter for :math:`a`.

    When :math:`a` is an integer, `gamma` reduces to the Erlang
    distribution, and when :math:`a=1` to the exponential distribution.

    Gamma distributions are sometimes parameterized with two variables,
    with a probability density function of:

    .. math::

        f(x, \\alpha, \\beta) =
        \\frac{\\beta^\\alpha x^{\\alpha - 1} e^{-\\beta x }}{\\Gamma(\\alpha)}

    Note that this parameterization is equivalent to the above, with
    ``scale = 1 / beta``.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('a', False, (0, np.inf), (False, False))]

    def _rvs(self, a, size=None, random_state=None):
        return random_state.standard_gamma(a, size)

    def _pdf(self, x, a):
        return np.exp(self._logpdf(x, a))

    def _logpdf(self, x, a):
        return sc.xlogy(a - 1.0, x) - x - sc.gammaln(a)

    def _cdf(self, x, a):
        return sc.gammainc(a, x)

    def _sf(self, x, a):
        return sc.gammaincc(a, x)

    def _ppf(self, q, a):
        return sc.gammaincinv(a, q)

    def _isf(self, q, a):
        return sc.gammainccinv(a, q)

    def _stats(self, a):
        return (a, a, 2.0 / np.sqrt(a), 6.0 / a)

    def _entropy(self, a):

        def regular_formula(a):
            return sc.psi(a) * (1 - a) + a + sc.gammaln(a)

        def asymptotic_formula(a):
            return 0.5 * (1.0 + np.log(2 * np.pi) + np.log(a)) - 1 / (3 * a) - a ** (-2.0) / 12 - a ** (-3.0) / 90 + a ** (-4.0) / 120
        return _lazywhere(a < 250, (a,), regular_formula, f2=asymptotic_formula)

    def _fitstart(self, data):
        if isinstance(data, CensoredData):
            data = data._uncensor()
        sk = _skew(data)
        a = 4 / (1e-08 + sk ** 2)
        return super()._fitstart(data, args=(a,))

    @extend_notes_in_docstring(rv_continuous, notes="        When the location is fixed by using the argument `floc`\n        and `method='MLE'`, this\n        function uses explicit formulas or solves a simpler numerical\n        problem than the full ML optimization problem.  So in that case,\n        the `optimizer`, `loc` and `scale` arguments are ignored.\n        \n\n")
    def fit(self, data, *args, **kwds):
        floc = kwds.get('floc', None)
        method = kwds.get('method', 'mle')
        if isinstance(data, CensoredData) or floc is None or method.lower() == 'mm':
            return super().fit(data, *args, **kwds)
        kwds.pop('floc', None)
        f0 = _get_fixed_fit_value(kwds, ['f0', 'fa', 'fix_a'])
        fscale = kwds.pop('fscale', None)
        _remove_optimizer_parameters(kwds)
        if f0 is not None and fscale is not None:
            raise ValueError('All parameters fixed. There is nothing to optimize.')
        data = np.asarray(data)
        if not np.isfinite(data).all():
            raise ValueError('The data contains non-finite values.')
        if np.any(data <= floc):
            raise FitDataError('gamma', lower=floc, upper=np.inf)
        if floc != 0:
            data = data - floc
        xbar = data.mean()
        if fscale is None:
            if f0 is not None:
                a = f0
            else:
                s = np.log(xbar) - np.log(data).mean()
                aest = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
                xa = aest * (1 - 0.4)
                xb = aest * (1 + 0.4)
                a = optimize.brentq(lambda a: np.log(a) - sc.digamma(a) - s, xa, xb, disp=0)
            scale = xbar / a
        else:
            c = np.log(data).mean() - np.log(fscale)
            a = _digammainv(c)
            scale = fscale
        return (a, floc, scale)