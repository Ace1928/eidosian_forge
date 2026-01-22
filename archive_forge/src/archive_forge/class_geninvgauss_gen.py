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
class geninvgauss_gen(rv_continuous):
    """A Generalized Inverse Gaussian continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `geninvgauss` is:

    .. math::

        f(x, p, b) = x^{p-1} \\exp(-b (x + 1/x) / 2) / (2 K_p(b))

    where `x > 0`, `p` is a real number and `b > 0`\\([1]_).
    :math:`K_p` is the modified Bessel function of second kind of order `p`
    (`scipy.special.kv`).

    %(after_notes)s

    The inverse Gaussian distribution `stats.invgauss(mu)` is a special case of
    `geninvgauss` with `p = -1/2`, `b = 1 / mu` and `scale = mu`.

    Generating random variates is challenging for this distribution. The
    implementation is based on [2]_.

    References
    ----------
    .. [1] O. Barndorff-Nielsen, P. Blaesild, C. Halgreen, "First hitting time
       models for the generalized inverse gaussian distribution",
       Stochastic Processes and their Applications 7, pp. 49--54, 1978.

    .. [2] W. Hoermann and J. Leydold, "Generating generalized inverse Gaussian
       random variates", Statistics and Computing, 24(4), p. 547--557, 2014.

    %(example)s

    """

    def _argcheck(self, p, b):
        return (p == p) & (b > 0)

    def _shape_info(self):
        ip = _ShapeInfo('p', False, (-np.inf, np.inf), (False, False))
        ib = _ShapeInfo('b', False, (0, np.inf), (False, False))
        return [ip, ib]

    def _logpdf(self, x, p, b):

        def logpdf_single(x, p, b):
            return _stats.geninvgauss_logpdf(x, p, b)
        logpdf_single = np.vectorize(logpdf_single, otypes=[np.float64])
        z = logpdf_single(x, p, b)
        if np.isnan(z).any():
            msg = 'Infinite values encountered in scipy.special.kve(p, b). Values replaced by NaN to avoid incorrect results.'
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return z

    def _pdf(self, x, p, b):
        return np.exp(self._logpdf(x, p, b))

    def _cdf(self, x, *args):
        _a, _b = self._get_support(*args)

        def _cdf_single(x, *args):
            p, b = args
            user_data = np.array([p, b], float).ctypes.data_as(ctypes.c_void_p)
            llc = LowLevelCallable.from_cython(_stats, '_geninvgauss_pdf', user_data)
            return integrate.quad(llc, _a, x)[0]
        _cdf_single = np.vectorize(_cdf_single, otypes=[np.float64])
        return _cdf_single(x, *args)

    def _logquasipdf(self, x, p, b):
        return _lazywhere(x > 0, (x, p, b), lambda x, p, b: (p - 1) * np.log(x) - b * (x + 1 / x) / 2, -np.inf)

    def _rvs(self, p, b, size=None, random_state=None):
        if np.isscalar(p) and np.isscalar(b):
            out = self._rvs_scalar(p, b, size, random_state)
        elif p.size == 1 and b.size == 1:
            out = self._rvs_scalar(p.item(), b.item(), size, random_state)
        else:
            p, b = np.broadcast_arrays(p, b)
            shp, bc = _check_shape(p.shape, size)
            numsamples = int(np.prod(shp))
            out = np.empty(size)
            it = np.nditer([p, b], flags=['multi_index'], op_flags=[['readonly'], ['readonly']])
            while not it.finished:
                idx = tuple((it.multi_index[j] if not bc[j] else slice(None) for j in range(-len(size), 0)))
                out[idx] = self._rvs_scalar(it[0], it[1], numsamples, random_state).reshape(shp)
                it.iternext()
        if size == ():
            out = out.item()
        return out

    def _rvs_scalar(self, p, b, numsamples, random_state):
        invert_res = False
        if not numsamples:
            numsamples = 1
        if p < 0:
            p = -p
            invert_res = True
        m = self._mode(p, b)
        ratio_unif = True
        if p >= 1 or b > 1:
            mode_shift = True
        elif b >= min(0.5, 2 * np.sqrt(1 - p) / 3):
            mode_shift = False
        else:
            ratio_unif = False
        size1d = tuple(np.atleast_1d(numsamples))
        N = np.prod(size1d)
        x = np.zeros(N)
        simulated = 0
        if ratio_unif:
            if mode_shift:
                a2 = -2 * (p + 1) / b - m
                a1 = 2 * m * (p - 1) / b - 1
                p1 = a1 - a2 ** 2 / 3
                q1 = 2 * a2 ** 3 / 27 - a2 * a1 / 3 + m
                phi = np.arccos(-q1 * np.sqrt(-27 / p1 ** 3) / 2)
                s1 = -np.sqrt(-4 * p1 / 3)
                root1 = s1 * np.cos(phi / 3 + np.pi / 3) - a2 / 3
                root2 = -s1 * np.cos(phi / 3) - a2 / 3
                lm = self._logquasipdf(m, p, b)
                d1 = self._logquasipdf(root1, p, b) - lm
                d2 = self._logquasipdf(root2, p, b) - lm
                vmin = (root1 - m) * np.exp(0.5 * d1)
                vmax = (root2 - m) * np.exp(0.5 * d2)
                umax = 1

                def logqpdf(x):
                    return self._logquasipdf(x, p, b) - lm
                c = m
            else:
                umax = np.exp(0.5 * self._logquasipdf(m, p, b))
                xplus = (1 + p + np.sqrt((1 + p) ** 2 + b ** 2)) / b
                vmin = 0
                vmax = xplus * np.exp(0.5 * self._logquasipdf(xplus, p, b))
                c = 0

                def logqpdf(x):
                    return self._logquasipdf(x, p, b)
            if vmin >= vmax:
                raise ValueError('vmin must be smaller than vmax.')
            if umax <= 0:
                raise ValueError('umax must be positive.')
            i = 1
            while simulated < N:
                k = N - simulated
                u = umax * random_state.uniform(size=k)
                v = random_state.uniform(size=k)
                v = vmin + (vmax - vmin) * v
                rvs = v / u + c
                accept = 2 * np.log(u) <= logqpdf(rvs)
                num_accept = np.sum(accept)
                if num_accept > 0:
                    x[simulated:simulated + num_accept] = rvs[accept]
                    simulated += num_accept
                if simulated == 0 and i * N >= 50000:
                    msg = f'Not a single random variate could be generated in {i * N} attempts. Sampling does not appear to work for the provided parameters.'
                    raise RuntimeError(msg)
                i += 1
        else:
            x0 = b / (1 - p)
            xs = np.max((x0, 2 / b))
            k1 = np.exp(self._logquasipdf(m, p, b))
            A1 = k1 * x0
            if x0 < 2 / b:
                k2 = np.exp(-b)
                if p > 0:
                    A2 = k2 * ((2 / b) ** p - x0 ** p) / p
                else:
                    A2 = k2 * np.log(2 / b ** 2)
            else:
                k2, A2 = (0, 0)
            k3 = xs ** (p - 1)
            A3 = 2 * k3 * np.exp(-xs * b / 2) / b
            A = A1 + A2 + A3
            while simulated < N:
                k = N - simulated
                h, rvs = (np.zeros(k), np.zeros(k))
                u = random_state.uniform(size=k)
                v = A * random_state.uniform(size=k)
                cond1 = v <= A1
                cond2 = np.logical_not(cond1) & (v <= A1 + A2)
                cond3 = np.logical_not(cond1 | cond2)
                rvs[cond1] = x0 * v[cond1] / A1
                h[cond1] = k1
                if p > 0:
                    rvs[cond2] = (x0 ** p + (v[cond2] - A1) * p / k2) ** (1 / p)
                else:
                    rvs[cond2] = b * np.exp((v[cond2] - A1) * np.exp(b))
                h[cond2] = k2 * rvs[cond2] ** (p - 1)
                z = np.exp(-xs * b / 2) - b * (v[cond3] - A1 - A2) / (2 * k3)
                rvs[cond3] = -2 / b * np.log(z)
                h[cond3] = k3 * np.exp(-rvs[cond3] * b / 2)
                accept = np.log(u * h) <= self._logquasipdf(rvs, p, b)
                num_accept = sum(accept)
                if num_accept > 0:
                    x[simulated:simulated + num_accept] = rvs[accept]
                    simulated += num_accept
        rvs = np.reshape(x, size1d)
        if invert_res:
            rvs = 1 / rvs
        return rvs

    def _mode(self, p, b):
        if p < 1:
            return b / (np.sqrt((p - 1) ** 2 + b ** 2) + 1 - p)
        else:
            return (np.sqrt((1 - p) ** 2 + b ** 2) - (1 - p)) / b

    def _munp(self, n, p, b):
        num = sc.kve(p + n, b)
        denom = sc.kve(p, b)
        inf_vals = np.isinf(num) | np.isinf(denom)
        if inf_vals.any():
            msg = 'Infinite values encountered in the moment calculation involving scipy.special.kve. Values replaced by NaN to avoid incorrect results.'
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
            m = np.full_like(num, np.nan, dtype=np.float64)
            m[~inf_vals] = num[~inf_vals] / denom[~inf_vals]
        else:
            m = num / denom
        return m