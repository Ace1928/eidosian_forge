from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
def trapezoid(y, x=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\\int_t y(t) dt =
    \\int_t y(t) \\left.\\frac{dx}{dt}\\right|_{x=x(t)} dt`.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapezoid : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    See Also
    --------
    cumulative_trapezoid, simpson, romb

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.

    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    Use the trapezoidal rule on evenly spaced points:

    >>> import numpy as np
    >>> from scipy import integrate
    >>> integrate.trapezoid([1, 2, 3])
    4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> integrate.trapezoid([1, 2, 3], dx=2)
    8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\\int_0^1 x^2 = 1/3` using:

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> integrate.trapezoid(y, x)
    0.33340274885464394

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> integrate.trapezoid(np.cos(theta), x=np.sin(theta))
    3.141571941375841

    ``trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> integrate.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> integrate.trapezoid(a, axis=1)
    array([2.,  8.])
    """
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret