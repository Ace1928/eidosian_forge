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
class skewnorm_gen(rv_continuous):
    """A skew-normal random variable.

    %(before_notes)s

    Notes
    -----
    The pdf is::

        skewnorm.pdf(x, a) = 2 * norm.pdf(x) * norm.cdf(a*x)

    `skewnorm` takes a real number :math:`a` as a skewness parameter
    When ``a = 0`` the distribution is identical to a normal distribution
    (`norm`). `rvs` implements the method of [1]_.

    %(after_notes)s

    %(example)s

    References
    ----------
    .. [1] A. Azzalini and A. Capitanio (1999). Statistical applications of
        the multivariate skew-normal distribution. J. Roy. Statist. Soc.,
        B 61, 579-602. :arxiv:`0911.2093`

    """

    def _argcheck(self, a):
        return np.isfinite(a)

    def _shape_info(self):
        return [_ShapeInfo('a', False, (-np.inf, np.inf), (False, False))]

    def _pdf(self, x, a):
        return _lazywhere(a == 0, (x, a), lambda x, a: _norm_pdf(x), f2=lambda x, a: 2.0 * _norm_pdf(x) * _norm_cdf(a * x))

    def _logpdf(self, x, a):
        return _lazywhere(a == 0, (x, a), lambda x, a: _norm_logpdf(x), f2=lambda x, a: np.log(2) + _norm_logpdf(x) + _norm_logcdf(a * x))

    def _cdf(self, x, a):
        a = np.atleast_1d(a)
        cdf = _boost._skewnorm_cdf(x, 0, 1, a)
        a = np.broadcast_to(a, cdf.shape)
        i_small_cdf = (cdf < 1e-06) & (a > 0)
        cdf[i_small_cdf] = super()._cdf(x[i_small_cdf], a[i_small_cdf])
        return np.clip(cdf, 0, 1)

    def _ppf(self, x, a):
        return _boost._skewnorm_ppf(x, 0, 1, a)

    def _sf(self, x, a):
        return self._cdf(-x, -a)

    def _isf(self, x, a):
        return _boost._skewnorm_isf(x, 0, 1, a)

    def _rvs(self, a, size=None, random_state=None):
        u0 = random_state.normal(size=size)
        v = random_state.normal(size=size)
        d = a / np.sqrt(1 + a ** 2)
        u1 = d * u0 + v * np.sqrt(1 - d ** 2)
        return np.where(u0 >= 0, u1, -u1)

    def _stats(self, a, moments='mvsk'):
        output = [None, None, None, None]
        const = np.sqrt(2 / np.pi) * a / np.sqrt(1 + a ** 2)
        if 'm' in moments:
            output[0] = const
        if 'v' in moments:
            output[1] = 1 - const ** 2
        if 's' in moments:
            output[2] = (4 - np.pi) / 2 * (const / np.sqrt(1 - const ** 2)) ** 3
        if 'k' in moments:
            output[3] = 2 * (np.pi - 3) * (const ** 4 / (1 - const ** 2) ** 2)
        return output

    @cached_property
    def _skewnorm_odd_moments(self):
        skewnorm_odd_moments = {1: Polynomial([1]), 3: Polynomial([3, -1]), 5: Polynomial([15, -10, 3]), 7: Polynomial([105, -105, 63, -15]), 9: Polynomial([945, -1260, 1134, -540, 105]), 11: Polynomial([10395, -17325, 20790, -14850, 5775, -945]), 13: Polynomial([135135, -270270, 405405, -386100, 225225, -73710, 10395]), 15: Polynomial([2027025, -4729725, 8513505, -10135125, 7882875, -3869775, 1091475, -135135]), 17: Polynomial([34459425, -91891800, 192972780, -275675400, 268017750, -175429800, 74220300, -18378360, 2027025]), 19: Polynomial([654729075, -1964187225, 4714049340, -7856748900, 9166207050, -7499623950, 4230557100, -1571349780, 346621275, -34459425])}
        return skewnorm_odd_moments

    def _munp(self, order, a):
        if order & 1:
            if order > 19:
                raise NotImplementedError('skewnorm noncentral moments not implemented for odd orders greater than 19.')
            delta = a / np.sqrt(1 + a ** 2)
            return delta * self._skewnorm_odd_moments[order](delta ** 2) * _SQRT_2_OVER_PI
        else:
            return sc.gamma((order + 1) / 2) * 2 ** (order / 2) / _SQRT_PI

    @extend_notes_in_docstring(rv_continuous, notes="        If ``method='mm'``, parameters fixed by the user are respected, and the\n        remaining parameters are used to match distribution and sample moments\n        where possible. For example, if the user fixes the location with\n        ``floc``, the parameters will only match the distribution skewness and\n        variance to the sample skewness and variance; no attempt will be made\n        to match the means or minimize a norm of the errors.\n        Note that the maximum possible skewness magnitude of a\n        `scipy.stats.skewnorm` distribution is approximately 0.9952717; if the\n        magnitude of the data's sample skewness exceeds this, the returned\n        shape parameter ``a`` will be infinite.\n        \n\n")
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        if isinstance(data, CensoredData):
            if data.num_censored() == 0:
                data = data._uncensor()
            else:
                return super().fit(data, *args, **kwds)
        data, fa, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        method = kwds.get('method', 'mle').lower()

        def skew_d(d):
            return (4 - np.pi) / 2 * ((d * np.sqrt(2 / np.pi)) ** 3 / (1 - 2 * d ** 2 / np.pi) ** (3 / 2))

        def d_skew(skew):
            s_23 = np.abs(skew) ** (2 / 3)
            return np.sign(skew) * np.sqrt(np.pi / 2 * s_23 / (s_23 + ((4 - np.pi) / 2) ** (2 / 3)))
        if method == 'mm':
            a, loc, scale = (None, None, None)
        else:
            a = args[0] if len(args) else None
            loc = kwds.pop('loc', None)
            scale = kwds.pop('scale', None)
        if fa is None and a is None:
            s = stats.skew(data)
            if method == 'mle':
                s = np.clip(s, -0.99, 0.99)
            else:
                s_max = skew_d(1)
                s = np.clip(s, -s_max, s_max)
            d = d_skew(s)
            with np.errstate(divide='ignore'):
                a = np.sqrt(np.divide(d ** 2, 1 - d ** 2)) * np.sign(s)
        else:
            a = fa if fa is not None else a
            d = a / np.sqrt(1 + a ** 2)
        if fscale is None and scale is None:
            v = np.var(data)
            scale = np.sqrt(v / (1 - 2 * d ** 2 / np.pi))
        elif fscale is not None:
            scale = fscale
        if floc is None and loc is None:
            m = np.mean(data)
            loc = m - scale * d * np.sqrt(2 / np.pi)
        elif floc is not None:
            loc = floc
        if method == 'mm':
            return (a, loc, scale)
        else:
            return super().fit(data, a, loc=loc, scale=scale, **kwds)