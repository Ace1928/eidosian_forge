from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
class rv_continuous(rv_generic):
    """A generic continuous random variable class meant for subclassing.

    `rv_continuous` is a base class to construct specific distribution classes
    and instances for continuous random variables. It cannot be used
    directly as a distribution.

    Parameters
    ----------
    momtype : int, optional
        The type of generic moment calculation to use: 0 for pdf, 1 (default)
        for ppf.
    a : float, optional
        Lower bound of the support of the distribution, default is minus
        infinity.
    b : float, optional
        Upper bound of the support of the distribution, default is plus
        infinity.
    xtol : float, optional
        The tolerance for fixed point calculation for generic ppf.
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pdf
    logpdf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    fit
    fit_loc_scale
    nnlf
    support

    Notes
    -----
    Public methods of an instance of a distribution class (e.g., ``pdf``,
    ``cdf``) check their arguments and pass valid arguments to private,
    computational methods (``_pdf``, ``_cdf``). For ``pdf(x)``, ``x`` is valid
    if it is within the support of the distribution.
    Whether a shape parameter is valid is decided by an ``_argcheck`` method
    (which defaults to checking that its arguments are strictly positive.)

    **Subclassing**

    New random variables can be defined by subclassing the `rv_continuous` class
    and re-defining at least the ``_pdf`` or the ``_cdf`` method (normalized
    to location 0 and scale 1).

    If positive argument checking is not correct for your RV
    then you will also need to re-define the ``_argcheck`` method.

    For most of the scipy.stats distributions, the support interval doesn't
    depend on the shape parameters. ``x`` being in the support interval is
    equivalent to ``self.a <= x <= self.b``.  If either of the endpoints of
    the support do depend on the shape parameters, then
    i) the distribution must implement the ``_get_support`` method; and
    ii) those dependent endpoints must be omitted from the distribution's
    call to the ``rv_continuous`` initializer.

    Correct, but potentially slow defaults exist for the remaining
    methods but for speed and/or accuracy you can over-ride::

      _logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf

    The default method ``_rvs`` relies on the inverse of the cdf, ``_ppf``,
    applied to a uniform random variate. In order to generate random variates
    efficiently, either the default ``_ppf`` needs to be overwritten (e.g.
    if the inverse cdf can expressed in an explicit form) or a sampling
    method needs to be implemented in a custom ``_rvs`` method.

    If possible, you should override ``_isf``, ``_sf`` or ``_logsf``.
    The main reason would be to improve numerical accuracy: for example,
    the survival function ``_sf`` is computed as ``1 - _cdf`` which can
    result in loss of precision if ``_cdf(x)`` is close to one.

    **Methods that can be overwritten by subclasses**
    ::

      _rvs
      _pdf
      _cdf
      _sf
      _ppf
      _isf
      _stats
      _munp
      _entropy
      _argcheck
      _get_support

    There are additional (internal and private) generic methods that can
    be useful for cross-checking and for debugging, but might work in all
    cases when directly called.

    A note on ``shapes``: subclasses need not specify them explicitly. In this
    case, `shapes` will be automatically deduced from the signatures of the
    overridden methods (`pdf`, `cdf` etc).
    If, for some reason, you prefer to avoid relying on introspection, you can
    specify ``shapes`` explicitly as an argument to the instance constructor.


    **Frozen Distributions**

    Normally, you must provide shape parameters (and, optionally, location and
    scale parameters to each call of a method of a distribution.

    Alternatively, the object may be called (as a function) to fix the shape,
    location, and scale parameters returning a "frozen" continuous RV object:

    rv = generic(<shape(s)>, loc=0, scale=1)
        `rv_frozen` object with the same methods but holding the given shape,
        location, and scale fixed

    **Statistics**

    Statistics are computed using numerical integration by default.
    For speed you can redefine this using ``_stats``:

     - take shape parameters and return mu, mu2, g1, g2
     - If you can't compute one of these, return it as None
     - Can also be defined with a keyword argument ``moments``, which is a
       string composed of "m", "v", "s", and/or "k".
       Only the components appearing in string should be computed and
       returned in the order "m", "v", "s", or "k"  with missing values
       returned as None.

    Alternatively, you can override ``_munp``, which takes ``n`` and shape
    parameters and returns the n-th non-central moment of the distribution.

    **Deepcopying / Pickling**

    If a distribution or frozen distribution is deepcopied (pickled/unpickled,
    etc.), any underlying random number generator is deepcopied with it. An
    implication is that if a distribution relies on the singleton RandomState
    before copying, it will rely on a copy of that random state after copying,
    and ``np.random.seed`` will no longer control the state.

    Examples
    --------
    To create a new Gaussian distribution, we would do the following:

    >>> from scipy.stats import rv_continuous
    >>> class gaussian_gen(rv_continuous):
    ...     "Gaussian distribution"
    ...     def _pdf(self, x):
    ...         return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
    >>> gaussian = gaussian_gen(name='gaussian')

    ``scipy.stats`` distributions are *instances*, so here we subclass
    `rv_continuous` and create an instance. With this, we now have
    a fully functional distribution with all relevant methods automagically
    generated by the framework.

    Note that above we defined a standard normal distribution, with zero mean
    and unit variance. Shifting and scaling of the distribution can be done
    by using ``loc`` and ``scale`` parameters: ``gaussian.pdf(x, loc, scale)``
    essentially computes ``y = (x - loc) / scale`` and
    ``gaussian._pdf(y) / scale``.

    """

    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        super().__init__(seed)
        self._ctor_param = dict(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        if badvalue is None:
            badvalue = nan
        if name is None:
            name = 'Distribution'
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = -inf
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes
        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf], locscale_in='loc=0, scale=1', locscale_out='loc, scale')
        self._attach_methods()
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = 'An '
            else:
                hstr = 'A '
            longname = hstr + name
        if sys.flags.optimize < 2:
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname, docdict=docdict, discrete='continuous')
            else:
                dct = dict(distcont)
                self._construct_doc(docdict, dct.get(self.name))

    def __getstate__(self):
        dct = self.__dict__.copy()
        attrs = ['_parse_args', '_parse_args_stats', '_parse_args_rvs', '_cdfvec', '_ppfvec', 'vecentropy', 'generic_moment']
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        """
        Attaches dynamically created methods to the rv_continuous instance.
        """
        self._attach_argparser_methods()
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = vectorize(self._entropy, otypes='d')
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1
        if self.moment_type == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes='d')
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes='d')
        self.generic_moment.nin = self.numargs + 1

    def _updated_ctor_param(self):
        """Return the current version of _ctor_param, possibly updated by user.

        Used by freezing.
        Keep this in sync with the signature of __init__.
        """
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['xtol'] = self.xtol
        dct['badvalue'] = self.badvalue
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x,) + args) - q

    def _ppf_single(self, q, *args):
        factor = 10.0
        left, right = self._get_support(*args)
        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q, *args) > 0.0:
                left, right = (left * factor, left)
        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q, *args) < 0.0:
                left, right = (right, right * factor)
        return optimize.brentq(self._ppf_to_solve, left, right, args=(q,) + args, xtol=self.xtol)

    def _mom_integ0(self, x, m, *args):
        return x ** m * self.pdf(x, *args)

    def _mom0_sc(self, m, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._mom_integ0, _a, _b, args=(m,) + args)[0]

    def _mom_integ1(self, q, m, *args):
        return self.ppf(q, *args) ** m

    def _mom1_sc(self, m, *args):
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,) + args)[0]

    def _pdf(self, x, *args):
        return _derivative(self._cdf, x, dx=1e-05, args=args, order=5)

    def _logpdf(self, x, *args):
        p = self._pdf(x, *args)
        with np.errstate(divide='ignore'):
            return log(p)

    def _logpxf(self, x, *args):
        return self._logpdf(x, *args)

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        return integrate.quad(self._pdf, _a, x, args=args)[0]

    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)

    def pdf(self, x, *args, **kwds):
        """Probability density function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        putmask(output, 1 - cond0 + np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args + (scale,))
            scale, goodargs = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output

    def logpdf(self, x, *args, **kwds):
        """Log of the probability density function at x of the given RV.

        This uses a more numerically accurate calculation if available.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logpdf : array_like
            Log of the probability density function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        putmask(output, 1 - cond0 + np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args + (scale,))
            scale, goodargs = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, x, *args, **kwds):
        """
        Cumulative distribution function of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        cdf : ndarray
            Cumulative distribution function evaluated at `x`

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= np.asarray(_b)) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, 1 - cond0 + np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, x, *args, **kwds):
        """Log of the cumulative distribution function at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logcdf : array_like
            Log of the cumulative distribution function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1 - cond0) * (cond1 == cond1) + np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, x, *args, **kwds):
        """Survival function (1 - `cdf`) at x of the given RV.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        sf : array_like
            Survival function evaluated at x

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, 1 - cond0 + np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._sf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, x, *args, **kwds):
        """Log of the survival function of the given RV.

        Returns the log of the "survival function," defined as (1 - `cdf`),
        evaluated at `x`.

        Parameters
        ----------
        x : array_like
            quantiles
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        logsf : ndarray
            Log of the survival function evaluated at `x`.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        """Percent point function (inverse of `cdf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            lower tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)
        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (scale, loc))
            scale, loc, goodargs = (goodargs[-2], goodargs[-1], goodargs[:-2])
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        """Inverse survival function (inverse of `sf`) at q of the given RV.

        Parameters
        ----------
        q : array_like
            upper tail probability
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        x : ndarray or scalar
            Quantile corresponding to the upper tail probability q.

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 1)
        cond3 = cond0 & (q == 0)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)
        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (scale, loc))
            scale, loc, goodargs = (goodargs[-2], goodargs[-1], goodargs[:-2])
            place(output, cond, self._isf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError as e:
            raise ValueError('Not enough input arguments.') from e
        return (loc, scale, args)

    def _nnlf_and_penalty(self, x, args):
        """
        Compute the penalized negative log-likelihood for the
        "standardized" data (i.e. already shifted by loc and
        scaled by scale) for the shape parameters in `args`.

        `x` can be a 1D numpy array or a CensoredData instance.
        """
        if isinstance(x, CensoredData):
            xs = x._supported(*self._get_support(*args))
            n_bad = len(x) - len(xs)
            i1, i2 = xs._interval.T
            terms = [self._logpdf(xs._uncensored, *args), self._logcdf(xs._left, *args), self._logsf(xs._right, *args), np.log(self._delta_cdf(i1, i2, *args))]
        else:
            cond0 = ~self._support_mask(x, *args)
            n_bad = np.count_nonzero(cond0)
            if n_bad > 0:
                x = argsreduce(~cond0, x)[0]
            terms = [self._logpdf(x, *args)]
        totals, bad_counts = zip(*[_sum_finite(term) for term in terms])
        total = sum(totals)
        n_bad += sum(bad_counts)
        return -total + n_bad * _LOGXMAX * 100

    def _penalized_nnlf(self, theta, x):
        """Penalized negative loglikelihood function.

        i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        if isinstance(x, CensoredData):
            x = (x - loc) / scale
            n_log_scale = (len(x) - x.num_censored()) * log(scale)
        else:
            x = (x - loc) / scale
            n_log_scale = len(x) * log(scale)
        return self._nnlf_and_penalty(x, args) + n_log_scale

    def _fitstart(self, data, args=None):
        """Starting point for fit (shape arguments + loc + scale)."""
        if args is None:
            args = (1.0,) * self.numargs
        loc, scale = self._fit_loc_scale_support(data, *args)
        return args + (loc, scale)

    def _reduce_func(self, args, kwds, data=None):
        """
        Return the (possibly reduced) function to optimize in order to find MLE
        estimates for the .fit method.
        """
        shapes = []
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                key = 'f' + str(j)
                names = [key, 'f' + s, 'fix_' + s]
                val = _get_fixed_fit_value(kwds, names)
                if val is not None:
                    kwds[key] = val
        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])
        methods = {'mle', 'mm'}
        method = kwds.pop('method', 'mle').lower()
        if method == 'mm':
            n_params = len(shapes) + 2 - len(fixedn)
            exponents = np.arange(1, n_params + 1)[:, np.newaxis]
            data_moments = np.sum(data[None, :] ** exponents / len(data), axis=1)

            def objective(theta, x):
                return self._moment_error(theta, x, data_moments)
        elif method == 'mle':
            objective = self._penalized_nnlf
        else:
            raise ValueError("Method '{}' not available; must be one of {}".format(method, methods))
        if len(fixedn) == 0:
            func = objective
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError('All parameters fixed. There is nothing to optimize.')

            def restore(args, theta):
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return objective(newtheta, x)
        return (x0, func, restore, args)

    def _moment_error(self, theta, x, data_moments):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        dist_moments = np.array([self.moment(i + 1, *args, loc=loc, scale=scale) for i in range(len(data_moments))])
        if np.any(np.isnan(dist_moments)):
            raise ValueError("Method of moments encountered a non-finite distribution moment and cannot continue. Consider trying method='MLE'.")
        return (((data_moments - dist_moments) / np.maximum(np.abs(data_moments), 1e-08)) ** 2).sum()

    def fit(self, data, *args, **kwds):
        """
        Return estimates of shape (if applicable), location, and scale
        parameters from data. The default estimation method is Maximum
        Likelihood Estimation (MLE), but Method of Moments (MM)
        is also available.

        Starting estimates for the fit are given by input arguments;
        for any arguments not provided with starting estimates,
        ``self._fitstart(data)`` is called to generate such.

        One can hold some parameters fixed to specific values by passing in
        keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)
        and ``floc`` and ``fscale`` (for location and scale parameters,
        respectively).

        Parameters
        ----------
        data : array_like or `CensoredData` instance
            Data to use in estimating the distribution parameters.
        arg1, arg2, arg3,... : floats, optional
            Starting value(s) for any shape-characterizing arguments (those not
            provided will be determined by a call to ``_fitstart(data)``).
            No default value.
        **kwds : floats, optional
            - `loc`: initial guess of the distribution's location parameter.
            - `scale`: initial guess of the distribution's scale parameter.

            Special keyword arguments are recognized as holding certain
            parameters fixed:

            - f0...fn : hold respective shape parameters fixed.
              Alternatively, shape parameters to fix can be specified by name.
              For example, if ``self.shapes == "a, b"``, ``fa`` and ``fix_a``
              are equivalent to ``f0``, and ``fb`` and ``fix_b`` are
              equivalent to ``f1``.

            - floc : hold location parameter fixed to specified value.

            - fscale : hold scale parameter fixed to specified value.

            - optimizer : The optimizer to use.  The optimizer must take
              ``func`` and starting position as the first two arguments,
              plus ``args`` (for extra arguments to pass to the
              function to be optimized) and ``disp=0`` to suppress
              output as keyword arguments.

            - method : The method to use. The default is "MLE" (Maximum
              Likelihood Estimate); "MM" (Method of Moments)
              is also available.

        Raises
        ------
        TypeError, ValueError
            If an input is invalid
        `~scipy.stats.FitError`
            If fitting fails or the fit produced would be invalid

        Returns
        -------
        parameter_tuple : tuple of floats
            Estimates for any shape parameters (if applicable), followed by
            those for location and scale. For most random variables, shape
            statistics will be returned, but there are exceptions (e.g.
            ``norm``).

        Notes
        -----
        With ``method="MLE"`` (default), the fit is computed by minimizing
        the negative log-likelihood function. A large, finite penalty
        (rather than infinite negative log-likelihood) is applied for
        observations beyond the support of the distribution.

        With ``method="MM"``, the fit is computed by minimizing the L2 norm
        of the relative errors between the first *k* raw (about zero) data
        moments and the corresponding distribution moments, where *k* is the
        number of non-fixed parameters.
        More precisely, the objective function is::

            (((data_moments - dist_moments)
              / np.maximum(np.abs(data_moments), 1e-8))**2).sum()

        where the constant ``1e-8`` avoids division by zero in case of
        vanishing data moments. Typically, this error norm can be reduced to
        zero.
        Note that the standard method of moments can produce parameters for
        which some data are outside the support of the fitted distribution;
        this implementation does nothing to prevent this.

        For either method,
        the returned answer is not guaranteed to be globally optimal; it
        may only be locally optimal, or the optimization may fail altogether.
        If the data contain any of ``np.nan``, ``np.inf``, or ``-np.inf``,
        the `fit` method will raise a ``RuntimeError``.

        Examples
        --------

        Generate some data to fit: draw random variates from the `beta`
        distribution

        >>> from scipy.stats import beta
        >>> a, b = 1., 2.
        >>> x = beta.rvs(a, b, size=1000)

        Now we can fit all four parameters (``a``, ``b``, ``loc`` and
        ``scale``):

        >>> a1, b1, loc1, scale1 = beta.fit(x)

        We can also use some prior knowledge about the dataset: let's keep
        ``loc`` and ``scale`` fixed:

        >>> a1, b1, loc1, scale1 = beta.fit(x, floc=0, fscale=1)
        >>> loc1, scale1
        (0, 1)

        We can also keep shape parameters fixed by using ``f``-keywords. To
        keep the zero-th shape parameter ``a`` equal 1, use ``f0=1`` or,
        equivalently, ``fa=1``:

        >>> a1, b1, loc1, scale1 = beta.fit(x, fa=1, floc=0, fscale=1)
        >>> a1
        1

        Not all distributions return estimates for the shape parameters.
        ``norm`` for example just returns estimates for location and scale:

        >>> from scipy.stats import norm
        >>> x = norm.rvs(a, b, size=1000, random_state=123)
        >>> loc1, scale1 = norm.fit(x)
        >>> loc1, scale1
        (0.92087172783841631, 2.0015750750324668)
        """
        method = kwds.get('method', 'mle').lower()
        censored = isinstance(data, CensoredData)
        if censored:
            if method != 'mle':
                raise ValueError('For censored data, the method must be "MLE".')
            if data.num_censored() == 0:
                data = data._uncensored
                censored = False
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError('Too many input arguments.')
        if not censored:
            data = np.asarray(data).ravel()
            if not np.isfinite(data).all():
                raise ValueError('The data contains non-finite values.')
        start = [None] * 2
        if Narg < self.numargs or not ('loc' in kwds and 'scale' in kwds):
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds, data=data)
        optimizer = kwds.pop('optimizer', optimize.fmin)
        optimizer = _fit_determine_optimizer(optimizer)
        if kwds:
            raise TypeError('Unknown arguments: %s.' % kwds)
        vals = optimizer(func, x0, args=(data,), disp=0)
        obj = func(vals, data)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        loc, scale, shapes = self._unpack_loc_scale(vals)
        if not (np.all(self._argcheck(*shapes)) and scale > 0):
            raise FitError('Optimization converged to parameters that are outside the range allowed by the distribution.')
        if method == 'mm':
            if not np.isfinite(obj):
                raise FitError('Optimization failed: either a data moment or fitted distribution moment is non-finite.')
        return vals

    def _fit_loc_scale_support(self, data, *args):
        """Estimate loc and scale parameters from data accounting for support.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        if isinstance(data, CensoredData):
            data = data._uncensor()
        else:
            data = np.asarray(data)
        loc_hat, scale_hat = self.fit_loc_scale(data, *args)
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        a, b = (_a, _b)
        support_width = b - a
        if support_width <= 0:
            return (loc_hat, scale_hat)
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return (loc_hat, scale_hat)
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin
        if support_width < np.inf:
            loc_hat = data_a - a - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return (loc_hat, scale_hat)
        if a > -np.inf:
            return (data_a - a - margin, 1)
        elif b < np.inf:
            return (data_b - b + margin, 1)
        else:
            raise RuntimeError

    def fit_loc_scale(self, data, *args):
        """
        Estimate loc and scale parameters from data using 1st and 2nd moments.

        Parameters
        ----------
        data : array_like
            Data to fit.
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        Lhat : float
            Estimated location parameter for the data.
        Shat : float
            Estimated scale parameter for the data.

        """
        mu, mu2 = self.stats(*args, **{'moments': 'mv'})
        tmp = asarray(data)
        muhat = tmp.mean()
        mu2hat = tmp.var()
        Shat = sqrt(mu2hat / mu2)
        with np.errstate(invalid='ignore'):
            Lhat = muhat - Shat * mu
        if not np.isfinite(Lhat):
            Lhat = 0
        if not (np.isfinite(Shat) and 0 < Shat):
            Shat = 1
        return (Lhat, Shat)

    def _entropy(self, *args):

        def integ(x):
            val = self._pdf(x, *args)
            return entr(val)
        _a, _b = self._get_support(*args)
        with np.errstate(over='ignore'):
            h = integrate.quad(integ, _a, _b)[0]
        if not np.isnan(h):
            return h
        else:
            low, upp = self.ppf([1e-10, 1.0 - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            return integrate.quad(integ, lower, upper)[0]

    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds):
        """Calculate expected value of a function with respect to the
        distribution by numerical integration.

        The expected value of a function ``f(x)`` with respect to a
        distribution ``dist`` is defined as::

                    ub
            E[f(x)] = Integral(f(x) * dist.pdf(x)),
                    lb

        where ``ub`` and ``lb`` are arguments and ``x`` has the ``dist.pdf(x)``
        distribution. If the bounds ``lb`` and ``ub`` correspond to the
        support of the distribution, e.g. ``[-inf, inf]`` in the default
        case, then the integral is the unrestricted expectation of ``f(x)``.
        Also, the function ``f(x)`` may be defined such that ``f(x)`` is ``0``
        outside a finite interval in which case the expectation is
        calculated within the finite range ``[lb, ub]``.

        Parameters
        ----------
        func : callable, optional
            Function for which integral is calculated. Takes only one argument.
            The default is the identity mapping f(x) = x.
        args : tuple, optional
            Shape parameters of the distribution.
        loc : float, optional
            Location parameter (default=0).
        scale : float, optional
            Scale parameter (default=1).
        lb, ub : scalar, optional
            Lower and upper bound for integration. Default is set to the
            support of the distribution.
        conditional : bool, optional
            If True, the integral is corrected by the conditional probability
            of the integration interval.  The return value is the expectation
            of the function, conditional on being in the given interval.
            Default is False.

        Additional keyword arguments are passed to the integration routine.

        Returns
        -------
        expect : float
            The calculated expected value.

        Notes
        -----
        The integration behavior of this function is inherited from
        `scipy.integrate.quad`. Neither this function nor
        `scipy.integrate.quad` can verify whether the integral exists or is
        finite. For example ``cauchy(0).mean()`` returns ``np.nan`` and
        ``cauchy(0).expect()`` returns ``0.0``.

        Likewise, the accuracy of results is not verified by the function.
        `scipy.integrate.quad` is typically reliable for integrals that are
        numerically favorable, but it is not guaranteed to converge
        to a correct value for all possible intervals and integrands. This
        function is provided for convenience; for critical applications,
        check results against other integration methods.

        The function is not vectorized.

        Examples
        --------

        To understand the effect of the bounds of integration consider

        >>> from scipy.stats import expon
        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0)
        0.6321205588285578

        This is close to

        >>> expon(1).cdf(2.0) - expon(1).cdf(0.0)
        0.6321205588285577

        If ``conditional=True``

        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0, conditional=True)
        1.0000000000000002

        The slight deviation from 1 is due to numerical integration.

        The integrand can be treated as a complex-valued function
        by passing ``complex_func=True`` to `scipy.integrate.quad` .

        >>> import numpy as np
        >>> from scipy.stats import vonmises
        >>> res = vonmises(loc=2, kappa=1).expect(lambda x: np.exp(1j*x),
        ...                                       complex_func=True)
        >>> res
        (-0.18576377217422957+0.40590124735052263j)

        >>> np.angle(res)  # location of the (circular) distribution
        2.0

        """
        lockwds = {'loc': loc, 'scale': scale}
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        if func is None:

            def fun(x, *args):
                return x * self.pdf(x, *args, **lockwds)
        else:

            def fun(x, *args):
                return func(x) * self.pdf(x, *args, **lockwds)
        if lb is None:
            lb = loc + _a * scale
        if ub is None:
            ub = loc + _b * scale
        cdf_bounds = self.cdf([lb, ub], *args, **lockwds)
        invfac = cdf_bounds[1] - cdf_bounds[0]
        kwds['args'] = args
        alpha = 0.05
        inner_bounds = np.array([alpha, 1 - alpha])
        cdf_inner_bounds = cdf_bounds[0] + invfac * inner_bounds
        c, d = loc + self._ppf(cdf_inner_bounds, *args) * scale
        lbc = integrate.quad(fun, lb, c, **kwds)[0]
        cd = integrate.quad(fun, c, d, **kwds)[0]
        dub = integrate.quad(fun, d, ub, **kwds)[0]
        vals = lbc + cd + dub
        if conditional:
            vals /= invfac
        return np.array(vals)[()]

    def _param_info(self):
        shape_info = self._shape_info()
        loc_info = _ShapeInfo('loc', False, (-np.inf, np.inf), (False, False))
        scale_info = _ShapeInfo('scale', False, (0, np.inf), (False, False))
        param_info = shape_info + [loc_info, scale_info]
        return param_info

    def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
        """
        Compute CDF(x2) - CDF(x1).

        Where x1 is greater than the median, compute SF(x1) - SF(x2),
        otherwise compute CDF(x2) - CDF(x1).

        This function is only useful if `dist.sf(x, ...)` has an implementation
        that is numerically more accurate than `1 - dist.cdf(x, ...)`.
        """
        cdf1 = self.cdf(x1, *args, loc=loc, scale=scale)
        result = np.where(cdf1 > 0.5, self.sf(x1, *args, loc=loc, scale=scale) - self.sf(x2, *args, loc=loc, scale=scale), self.cdf(x2, *args, loc=loc, scale=scale) - cdf1)
        if result.ndim == 0:
            result = result[()]
        return result