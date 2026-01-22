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
class powerlaw_gen(rv_continuous):
    """A power-function continuous random variable.

    %(before_notes)s

    See Also
    --------
    pareto

    Notes
    -----
    The probability density function for `powerlaw` is:

    .. math::

        f(x, a) = a x^{a-1}

    for :math:`0 \\le x \\le 1`, :math:`a > 0`.

    `powerlaw` takes ``a`` as a shape parameter for :math:`a`.

    %(after_notes)s

    For example, the support of `powerlaw` can be adjusted from the default
    interval ``[0, 1]`` to the interval ``[c, c+d]`` by setting ``loc=c`` and
    ``scale=d``. For a power-law distribution with infinite support, see
    `pareto`.

    `powerlaw` is a special case of `beta` with ``b=1``.

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('a', False, (0, np.inf), (False, False))]

    def _pdf(self, x, a):
        return a * x ** (a - 1.0)

    def _logpdf(self, x, a):
        return np.log(a) + sc.xlogy(a - 1, x)

    def _cdf(self, x, a):
        return x ** (a * 1.0)

    def _logcdf(self, x, a):
        return a * np.log(x)

    def _ppf(self, q, a):
        return pow(q, 1.0 / a)

    def _sf(self, p, a):
        return -sc.powm1(p, a)

    def _stats(self, a):
        return (a / (a + 1.0), a / (a + 2.0) / (a + 1.0) ** 2, -2.0 * ((a - 1.0) / (a + 3.0)) * np.sqrt((a + 2.0) / a), 6 * np.polyval([1, -1, -6, 2], a) / (a * (a + 3.0) * (a + 4)))

    def _entropy(self, a):
        return 1 - 1.0 / a - np.log(a)

    def _support_mask(self, x, a):
        return super()._support_mask(x, a) & ((x != 0) | (a >= 1))

    @_call_super_mom
    @extend_notes_in_docstring(rv_continuous, notes='        Notes specifically for ``powerlaw.fit``: If the location is a free\n        parameter and the value returned for the shape parameter is less than\n        one, the true maximum likelihood approaches infinity. This causes\n        numerical difficulties, and the resulting estimates are approximate.\n        \n\n')
    def fit(self, data, *args, **kwds):
        if kwds.pop('superfit', False):
            return super().fit(data, *args, **kwds)
        if len(np.unique(data)) == 1:
            return super().fit(data, *args, **kwds)
        data, fshape, floc, fscale = _check_fit_input_parameters(self, data, args, kwds)
        penalized_nllf_args = [data, (self._fitstart(data),)]
        penalized_nllf = self._reduce_func(penalized_nllf_args, {})[1]
        if floc is not None:
            if not data.min() > floc:
                raise FitDataError('powerlaw', 0, 1)
            if fscale is not None and (not data.max() <= floc + fscale):
                raise FitDataError('powerlaw', 0, 1)
        if fscale is not None:
            if fscale <= 0:
                raise ValueError('Negative or zero `fscale` is outside the range allowed by the distribution.')
            if fscale <= np.ptp(data):
                msg = '`fscale` must be greater than the range of data.'
                raise ValueError(msg)

        def get_shape(data, loc, scale):
            N = len(data)
            return -N / (np.sum(np.log(data - loc)) - N * np.log(scale))

        def get_scale(data, loc):
            return data.max() - loc
        if fscale is not None and floc is not None:
            return (get_shape(data, floc, fscale), floc, fscale)
        if fscale is not None:
            loc_lt1 = np.nextafter(data.min(), -np.inf)
            shape_lt1 = fshape or get_shape(data, loc_lt1, fscale)
            ll_lt1 = penalized_nllf((shape_lt1, loc_lt1, fscale), data)
            loc_gt1 = np.nextafter(data.max() - fscale, np.inf)
            shape_gt1 = fshape or get_shape(data, loc_gt1, fscale)
            ll_gt1 = penalized_nllf((shape_gt1, loc_gt1, fscale), data)
            if ll_lt1 < ll_gt1:
                return (shape_lt1, loc_lt1, fscale)
            else:
                return (shape_gt1, loc_gt1, fscale)
        if floc is not None:
            scale = get_scale(data, floc)
            shape = fshape or get_shape(data, floc, scale)
            return (shape, floc, scale)

        def fit_loc_scale_w_shape_lt_1():
            loc = np.nextafter(data.min(), -np.inf)
            if np.abs(loc) < np.finfo(loc.dtype).tiny:
                loc = np.sign(loc) * np.finfo(loc.dtype).tiny
            scale = np.nextafter(get_scale(data, loc), np.inf)
            shape = fshape or get_shape(data, loc, scale)
            return (shape, loc, scale)

        def dL_dScale(data, shape, scale):
            return -data.shape[0] * shape / scale

        def dL_dLocation(data, shape, loc):
            return (shape - 1) * np.sum(1 / (loc - data))

        def dL_dLocation_star(loc):
            scale = np.nextafter(get_scale(data, loc), -np.inf)
            shape = fshape or get_shape(data, loc, scale)
            return dL_dLocation(data, shape, loc)

        def fun_to_solve(loc):
            scale = np.nextafter(get_scale(data, loc), -np.inf)
            shape = fshape or get_shape(data, loc, scale)
            return dL_dScale(data, shape, scale) - dL_dLocation(data, shape, loc)

        def fit_loc_scale_w_shape_gt_1():
            rbrack = np.nextafter(data.min(), -np.inf)
            delta = data.min() - rbrack
            while dL_dLocation_star(rbrack) > 0:
                rbrack = data.min() - delta
                delta *= 2

            def interval_contains_root(lbrack, rbrack):
                return np.sign(fun_to_solve(lbrack)) != np.sign(fun_to_solve(rbrack))
            lbrack = rbrack - 1
            i = 1.0
            while not interval_contains_root(lbrack, rbrack) and lbrack != -np.inf:
                lbrack = data.min() - i
                i *= 2
            root = optimize.root_scalar(fun_to_solve, bracket=(lbrack, rbrack))
            loc = np.nextafter(root.root, -np.inf)
            scale = np.nextafter(get_scale(data, loc), np.inf)
            shape = fshape or get_shape(data, loc, scale)
            return (shape, loc, scale)
        if fshape is not None and fshape <= 1:
            return fit_loc_scale_w_shape_lt_1()
        elif fshape is not None and fshape > 1:
            return fit_loc_scale_w_shape_gt_1()
        fit_shape_lt1 = fit_loc_scale_w_shape_lt_1()
        ll_lt1 = self.nnlf(fit_shape_lt1, data)
        fit_shape_gt1 = fit_loc_scale_w_shape_gt_1()
        ll_gt1 = self.nnlf(fit_shape_gt1, data)
        if ll_lt1 <= ll_gt1 and fit_shape_lt1[0] <= 1:
            return fit_shape_lt1
        elif ll_lt1 > ll_gt1 and fit_shape_gt1[0] > 1:
            return fit_shape_gt1
        else:
            return super().fit(data, *args, **kwds)