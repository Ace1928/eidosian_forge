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
class pearson3_gen(rv_continuous):
    """A pearson type III continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `pearson3` is:

    .. math::

        f(x, \\kappa) = \\frac{|\\beta|}{\\Gamma(\\alpha)}
                       (\\beta (x - \\zeta))^{\\alpha - 1}
                       \\exp(-\\beta (x - \\zeta))

    where:

    .. math::

            \\beta = \\frac{2}{\\kappa}

            \\alpha = \\beta^2 = \\frac{4}{\\kappa^2}

            \\zeta = -\\frac{\\alpha}{\\beta} = -\\beta

    :math:`\\Gamma` is the gamma function (`scipy.special.gamma`).
    Pass the skew :math:`\\kappa` into `pearson3` as the shape parameter
    ``skew``.

    %(after_notes)s

    %(example)s

    References
    ----------
    R.W. Vogel and D.E. McMartin, "Probability Plot Goodness-of-Fit and
    Skewness Estimation Procedures for the Pearson Type 3 Distribution", Water
    Resources Research, Vol.27, 3149-3158 (1991).

    L.R. Salvosa, "Tables of Pearson's Type III Function", Ann. Math. Statist.,
    Vol.1, 191-198 (1930).

    "Using Modern Computing Tools to Fit the Pearson Type III Distribution to
    Aviation Loads Data", Office of Aviation Research (2003).

    """

    def _preprocess(self, x, skew):
        loc = 0.0
        scale = 1.0
        norm2pearson_transition = 1.6e-05
        ans, x, skew = np.broadcast_arrays(1.0, x, skew)
        ans = ans.copy()
        mask = np.absolute(skew) < norm2pearson_transition
        invmask = ~mask
        beta = 2.0 / (skew[invmask] * scale)
        alpha = (scale * beta) ** 2
        zeta = loc - alpha / beta
        transx = beta * (x[invmask] - zeta)
        return (ans, x, transx, mask, invmask, beta, alpha, zeta)

    def _argcheck(self, skew):
        return np.isfinite(skew)

    def _shape_info(self):
        return [_ShapeInfo('skew', False, (-np.inf, np.inf), (False, False))]

    def _stats(self, skew):
        m = 0.0
        v = 1.0
        s = skew
        k = 1.5 * skew ** 2
        return (m, v, s, k)

    def _pdf(self, x, skew):
        ans = np.exp(self._logpdf(x, skew))
        if ans.ndim == 0:
            if np.isnan(ans):
                return 0.0
            return ans
        ans[np.isnan(ans)] = 0.0
        return ans

    def _logpdf(self, x, skew):
        ans, x, transx, mask, invmask, beta, alpha, _ = self._preprocess(x, skew)
        ans[mask] = np.log(_norm_pdf(x[mask]))
        ans[invmask] = np.log(abs(beta)) + gamma.logpdf(transx, alpha)
        return ans

    def _cdf(self, x, skew):
        ans, x, transx, mask, invmask, _, alpha, _ = self._preprocess(x, skew)
        ans[mask] = _norm_cdf(x[mask])
        skew = np.broadcast_to(skew, invmask.shape)
        invmask1a = np.logical_and(invmask, skew > 0)
        invmask1b = skew[invmask] > 0
        ans[invmask1a] = gamma.cdf(transx[invmask1b], alpha[invmask1b])
        invmask2a = np.logical_and(invmask, skew < 0)
        invmask2b = skew[invmask] < 0
        ans[invmask2a] = gamma.sf(transx[invmask2b], alpha[invmask2b])
        return ans

    def _sf(self, x, skew):
        ans, x, transx, mask, invmask, _, alpha, _ = self._preprocess(x, skew)
        ans[mask] = _norm_sf(x[mask])
        skew = np.broadcast_to(skew, invmask.shape)
        invmask1a = np.logical_and(invmask, skew > 0)
        invmask1b = skew[invmask] > 0
        ans[invmask1a] = gamma.sf(transx[invmask1b], alpha[invmask1b])
        invmask2a = np.logical_and(invmask, skew < 0)
        invmask2b = skew[invmask] < 0
        ans[invmask2a] = gamma.cdf(transx[invmask2b], alpha[invmask2b])
        return ans

    def _rvs(self, skew, size=None, random_state=None):
        skew = np.broadcast_to(skew, size)
        ans, _, _, mask, invmask, beta, alpha, zeta = self._preprocess([0], skew)
        nsmall = mask.sum()
        nbig = mask.size - nsmall
        ans[mask] = random_state.standard_normal(nsmall)
        ans[invmask] = random_state.standard_gamma(alpha, nbig) / beta + zeta
        if size == ():
            ans = ans[0]
        return ans

    def _ppf(self, q, skew):
        ans, q, _, mask, invmask, beta, alpha, zeta = self._preprocess(q, skew)
        ans[mask] = _norm_ppf(q[mask])
        q = q[invmask]
        q[beta < 0] = 1 - q[beta < 0]
        ans[invmask] = sc.gammaincinv(alpha, q) / beta + zeta
        return ans

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes="        Note that method of moments (`method='MM'`) is not\n        available for this distribution.\n\n")
    def fit(self, data, *args, **kwds):
        if kwds.get('method', None) == 'MM':
            raise NotImplementedError("Fit `method='MM'` is not available for the Pearson3 distribution. Please try the default `method='MLE'`.")
        else:
            return super(type(self), self).fit(data, *args, **kwds)