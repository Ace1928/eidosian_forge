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
class pareto_gen(rv_continuous):
    """A Pareto continuous random variable.

    %(before_notes)s

    Notes
    -----
    The probability density function for `pareto` is:

    .. math::

        f(x, b) = \\frac{b}{x^{b+1}}

    for :math:`x \\ge 1`, :math:`b > 0`.

    `pareto` takes ``b`` as a shape parameter for :math:`b`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo('b', False, (0, np.inf), (False, False))]

    def _pdf(self, x, b):
        return b * x ** (-b - 1)

    def _cdf(self, x, b):
        return 1 - x ** (-b)

    def _ppf(self, q, b):
        return pow(1 - q, -1.0 / b)

    def _sf(self, x, b):
        return x ** (-b)

    def _isf(self, q, b):
        return np.power(q, -1.0 / b)

    def _stats(self, b, moments='mv'):
        mu, mu2, g1, g2 = (None, None, None, None)
        if 'm' in moments:
            mask = b > 1
            bt = np.extract(mask, b)
            mu = np.full(np.shape(b), fill_value=np.inf)
            np.place(mu, mask, bt / (bt - 1.0))
        if 'v' in moments:
            mask = b > 2
            bt = np.extract(mask, b)
            mu2 = np.full(np.shape(b), fill_value=np.inf)
            np.place(mu2, mask, bt / (bt - 2.0) / (bt - 1.0) ** 2)
        if 's' in moments:
            mask = b > 3
            bt = np.extract(mask, b)
            g1 = np.full(np.shape(b), fill_value=np.nan)
            vals = 2 * (bt + 1.0) * np.sqrt(bt - 2.0) / ((bt - 3.0) * np.sqrt(bt))
            np.place(g1, mask, vals)
        if 'k' in moments:
            mask = b > 4
            bt = np.extract(mask, b)
            g2 = np.full(np.shape(b), fill_value=np.nan)
            vals = 6.0 * np.polyval([1.0, 1.0, -6, -2], bt) / np.polyval([1.0, -7.0, 12.0, 0.0], bt)
            np.place(g2, mask, vals)
        return (mu, mu2, g1, g2)

    def _entropy(self, c):
        return 1 + 1.0 / c - np.log(c)

    @_call_super_mom
    @inherit_docstring_from(rv_continuous)
    def fit(self, data, *args, **kwds):
        parameters = _check_fit_input_parameters(self, data, args, kwds)
        data, fshape, floc, fscale = parameters
        if floc is not None and np.min(data) - floc < (fscale or 0):
            raise FitDataError('pareto', lower=1, upper=np.inf)
        ndata = data.shape[0]

        def get_shape(scale, location):
            return ndata / np.sum(np.log((data - location) / scale))
        if floc is fscale is None:

            def dL_dScale(shape, scale):
                return ndata * shape / scale

            def dL_dLocation(shape, location):
                return (shape + 1) * np.sum(1 / (data - location))

            def fun_to_solve(scale):
                location = np.min(data) - scale
                shape = fshape or get_shape(scale, location)
                return dL_dLocation(shape, location) - dL_dScale(shape, scale)

            def interval_contains_root(lbrack, rbrack):
                return np.sign(fun_to_solve(lbrack)) != np.sign(fun_to_solve(rbrack))
            brack_start = float(kwds.get('scale', 1))
            lbrack, rbrack = (brack_start / 2, brack_start * 2)
            while not interval_contains_root(lbrack, rbrack) and (lbrack > 0 or rbrack < np.inf):
                lbrack /= 2
                rbrack *= 2
            res = root_scalar(fun_to_solve, bracket=[lbrack, rbrack])
            if res.converged:
                scale = res.root
                loc = np.min(data) - scale
                shape = fshape or get_shape(scale, loc)
                if not scale + loc < np.min(data):
                    scale = np.min(data) - loc
                    scale = np.nextafter(scale, 0)
                return (shape, loc, scale)
            else:
                return super().fit(data, **kwds)
        elif floc is None:
            loc = np.min(data) - fscale
        else:
            loc = floc
        scale = fscale or np.min(data) - loc
        shape = fshape or get_shape(scale, loc)
        return (shape, loc, scale)