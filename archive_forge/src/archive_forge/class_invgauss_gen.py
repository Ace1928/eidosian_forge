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
class invgauss_gen(rv_continuous):
    """An inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `invgauss` is:

    .. math::

        f(x, \\mu) = \\frac{1}{\\sqrt{2 \\pi x^3}}
                    \\exp(-\\frac{(x-\\mu)^2}{2 x \\mu^2})

    for :math:`x >= 0` and :math:`\\mu > 0`.

    `invgauss` takes ``mu`` as a shape parameter for :math:`\\mu`.

    %(after_notes)s

    %(example)s

    """
    _support_mask = rv_continuous._open_support_mask

    def _shape_info(self):
        return [_ShapeInfo('mu', False, (0, np.inf), (False, False))]

    def _rvs(self, mu, size=None, random_state=None):
        return random_state.wald(mu, 1.0, size=size)

    def _pdf(self, x, mu):
        return 1.0 / np.sqrt(2 * np.pi * x ** 3.0) * np.exp(-1.0 / (2 * x) * ((x - mu) / mu) ** 2)

    def _logpdf(self, x, mu):
        return -0.5 * np.log(2 * np.pi) - 1.5 * np.log(x) - ((x - mu) / mu) ** 2 / (2 * x)

    def _logcdf(self, x, mu):
        fac = 1 / np.sqrt(x)
        a = _norm_logcdf(fac * (x / mu - 1))
        b = 2 / mu + _norm_logcdf(-fac * (x / mu + 1))
        return a + np.log1p(np.exp(b - a))

    def _logsf(self, x, mu):
        fac = 1 / np.sqrt(x)
        a = _norm_logsf(fac * (x / mu - 1))
        b = 2 / mu + _norm_logcdf(-fac * (x + mu) / mu)
        return a + np.log1p(-np.exp(b - a))

    def _sf(self, x, mu):
        return np.exp(self._logsf(x, mu))

    def _cdf(self, x, mu):
        return np.exp(self._logcdf(x, mu))

    def _ppf(self, x, mu):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            x, mu = np.broadcast_arrays(x, mu)
            ppf = _boost._invgauss_ppf(x, mu, 1)
            i_wt = x > 0.5
            ppf[i_wt] = _boost._invgauss_isf(1 - x[i_wt], mu[i_wt], 1)
            i_nan = np.isnan(ppf)
            ppf[i_nan] = super()._ppf(x[i_nan], mu[i_nan])
        return ppf

    def _isf(self, x, mu):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            x, mu = np.broadcast_arrays(x, mu)
            isf = _boost._invgauss_isf(x, mu, 1)
            i_wt = x > 0.5
            isf[i_wt] = _boost._invgauss_ppf(1 - x[i_wt], mu[i_wt], 1)
            i_nan = np.isnan(isf)
            isf[i_nan] = super()._isf(x[i_nan], mu[i_nan])
        return isf

    def _stats(self, mu):
        return (mu, mu ** 3.0, 3 * np.sqrt(mu), 15 * mu)

    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        method = kwds.get('method', 'mle')
        if isinstance(data, CensoredData) or type(self) == wald_gen or method.lower() == 'mm':
            return super().fit(data, *args, **kwds)
        data, fshape_s, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        "\n        Source: Statistical Distributions, 3rd Edition. Evans, Hastings,\n        and Peacock (2000), Page 121. Their shape parameter is equivalent to\n        SciPy's with the conversion `fshape_s = fshape / scale`.\n\n        MLE formulas are not used in 3 conditions:\n        - `loc` is not fixed\n        - `mu` is fixed\n        These cases fall back on the superclass fit method.\n        - `loc` is fixed but translation results in negative data raises\n          a `FitDataError`.\n        "
        if floc is None or fshape_s is not None:
            return super().fit(data, *args, **kwds)
        elif np.any(data - floc < 0):
            raise FitDataError('invgauss', lower=0, upper=np.inf)
        else:
            data = data - floc
            fshape_n = np.mean(data)
            if fscale is None:
                fscale = len(data) / np.sum(data ** (-1) - fshape_n ** (-1))
            fshape_s = fshape_n / fscale
        return (fshape_s, floc, fscale)

    def _entropy(self, mu):
        """
        Ref.: https://moser-isi.ethz.ch/docs/papers/smos-2012-10.pdf (eq. 9)
        """
        a = 1.0 + np.log(2 * np.pi) + 3 * np.log(mu)
        r = 2 / mu
        b = sc._ufuncs._scaled_exp1(r) / r
        return 0.5 * a - 1.5 * b