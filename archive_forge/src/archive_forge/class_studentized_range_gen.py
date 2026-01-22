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
class studentized_range_gen(rv_continuous):
    """A studentized range continuous random variable.

    %(before_notes)s

    See Also
    --------
    t: Student's t distribution

    Notes
    -----
    The probability density function for `studentized_range` is:

    .. math::

         f(x; k, \\nu) = \\frac{k(k-1)\\nu^{\\nu/2}}{\\Gamma(\\nu/2)
                        2^{\\nu/2-1}} \\int_{0}^{\\infty} \\int_{-\\infty}^{\\infty}
                        s^{\\nu} e^{-\\nu s^2/2} \\phi(z) \\phi(sx + z)
                        [\\Phi(sx + z) - \\Phi(z)]^{k-2} \\,dz \\,ds

    for :math:`x ≥ 0`, :math:`k > 1`, and :math:`\\nu > 0`.

    `studentized_range` takes ``k`` for :math:`k` and ``df`` for :math:`\\nu`
    as shape parameters.

    When :math:`\\nu` exceeds 100,000, an asymptotic approximation (infinite
    degrees of freedom) is used to compute the cumulative distribution
    function [4]_ and probability distribution function.

    %(after_notes)s

    References
    ----------

    .. [1] "Studentized range distribution",
           https://en.wikipedia.org/wiki/Studentized_range_distribution
    .. [2] Batista, Ben Dêivide, et al. "Externally Studentized Normal Midrange
           Distribution." Ciência e Agrotecnologia, vol. 41, no. 4, 2017, pp.
           378-389., doi:10.1590/1413-70542017414047716.
    .. [3] Harter, H. Leon. "Tables of Range and Studentized Range." The Annals
           of Mathematical Statistics, vol. 31, no. 4, 1960, pp. 1122-1147.
           JSTOR, www.jstor.org/stable/2237810. Accessed 18 Feb. 2021.
    .. [4] Lund, R. E., and J. R. Lund. "Algorithm AS 190: Probabilities and
           Upper Quantiles for the Studentized Range." Journal of the Royal
           Statistical Society. Series C (Applied Statistics), vol. 32, no. 2,
           1983, pp. 204-210. JSTOR, www.jstor.org/stable/2347300. Accessed 18
           Feb. 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import studentized_range
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Calculate the first four moments:

    >>> k, df = 3, 10
    >>> mean, var, skew, kurt = studentized_range.stats(k, df, moments='mvsk')

    Display the probability density function (``pdf``):

    >>> x = np.linspace(studentized_range.ppf(0.01, k, df),
    ...                 studentized_range.ppf(0.99, k, df), 100)
    >>> ax.plot(x, studentized_range.pdf(x, k, df),
    ...         'r-', lw=5, alpha=0.6, label='studentized_range pdf')

    Alternatively, the distribution object can be called (as a function)
    to fix the shape, location and scale parameters. This returns a "frozen"
    RV object holding the given parameters fixed.

    Freeze the distribution and display the frozen ``pdf``:

    >>> rv = studentized_range(k, df)
    >>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    Check accuracy of ``cdf`` and ``ppf``:

    >>> vals = studentized_range.ppf([0.001, 0.5, 0.999], k, df)
    >>> np.allclose([0.001, 0.5, 0.999], studentized_range.cdf(vals, k, df))
    True

    Rather than using (``studentized_range.rvs``) to generate random variates,
    which is very slow for this distribution, we can approximate the inverse
    CDF using an interpolator, and then perform inverse transform sampling
    with this approximate inverse CDF.

    This distribution has an infinite but thin right tail, so we focus our
    attention on the leftmost 99.9 percent.

    >>> a, b = studentized_range.ppf([0, .999], k, df)
    >>> a, b
    0, 7.41058083802274

    >>> from scipy.interpolate import interp1d
    >>> rng = np.random.default_rng()
    >>> xs = np.linspace(a, b, 50)
    >>> cdf = studentized_range.cdf(xs, k, df)
    # Create an interpolant of the inverse CDF
    >>> ppf = interp1d(cdf, xs, fill_value='extrapolate')
    # Perform inverse transform sampling using the interpolant
    >>> r = ppf(rng.uniform(size=1000))

    And compare the histogram:

    >>> ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    >>> ax.legend(loc='best', frameon=False)
    >>> plt.show()

    """

    def _argcheck(self, k, df):
        return (k > 1) & (df > 0)

    def _shape_info(self):
        ik = _ShapeInfo('k', False, (1, np.inf), (False, False))
        idf = _ShapeInfo('df', False, (0, np.inf), (False, False))
        return [ik, idf]

    def _fitstart(self, data):
        return super()._fitstart(data, args=(2, 1))

    def _munp(self, K, k, df):
        cython_symbol = '_studentized_range_moment'
        _a, _b = self._get_support()

        def _single_moment(K, k, df):
            log_const = _stats._studentized_range_pdf_logconst(k, df)
            arg = [K, k, df, log_const]
            usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
            llc = LowLevelCallable.from_cython(_stats, cython_symbol, usr_data)
            ranges = [(-np.inf, np.inf), (0, np.inf), (_a, _b)]
            opts = dict(epsabs=1e-11, epsrel=1e-12)
            return integrate.nquad(llc, ranges=ranges, opts=opts)[0]
        ufunc = np.frompyfunc(_single_moment, 3, 1)
        return np.asarray(ufunc(K, k, df), dtype=np.float64)[()]

    def _pdf(self, x, k, df):

        def _single_pdf(q, k, df):
            if df < 100000:
                cython_symbol = '_studentized_range_pdf'
                log_const = _stats._studentized_range_pdf_logconst(k, df)
                arg = [q, k, df, log_const]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf), (0, np.inf)]
            else:
                cython_symbol = '_studentized_range_pdf_asymptotic'
                arg = [q, k]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf)]
            llc = LowLevelCallable.from_cython(_stats, cython_symbol, usr_data)
            opts = dict(epsabs=1e-11, epsrel=1e-12)
            return integrate.nquad(llc, ranges=ranges, opts=opts)[0]
        ufunc = np.frompyfunc(_single_pdf, 3, 1)
        return np.asarray(ufunc(x, k, df), dtype=np.float64)[()]

    def _cdf(self, x, k, df):

        def _single_cdf(q, k, df):
            if df < 100000:
                cython_symbol = '_studentized_range_cdf'
                log_const = _stats._studentized_range_cdf_logconst(k, df)
                arg = [q, k, df, log_const]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf), (0, np.inf)]
            else:
                cython_symbol = '_studentized_range_cdf_asymptotic'
                arg = [q, k]
                usr_data = np.array(arg, float).ctypes.data_as(ctypes.c_void_p)
                ranges = [(-np.inf, np.inf)]
            llc = LowLevelCallable.from_cython(_stats, cython_symbol, usr_data)
            opts = dict(epsabs=1e-11, epsrel=1e-12)
            return integrate.nquad(llc, ranges=ranges, opts=opts)[0]
        ufunc = np.frompyfunc(_single_cdf, 3, 1)
        return np.clip(np.asarray(ufunc(x, k, df), dtype=np.float64)[()], 0, 1)