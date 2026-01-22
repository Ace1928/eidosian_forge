import numpy as np
from ._fitpack_impl import bisplrep, bisplev, dblint  # noqa: F401
from . import _fitpack_impl as _impl
from ._bsplines import BSpline

    Compute the spline for the antiderivative (integral) of a given spline.

    Parameters
    ----------
    tck : BSpline instance or a tuple of (t, c, k)
        Spline whose antiderivative to compute
    n : int, optional
        Order of antiderivative to evaluate. Default: 1

    Returns
    -------
    BSpline instance or a tuple of (t2, c2, k2)
        Spline of order k2=k+n representing the antiderivative of the input
        spline.
        A tuple is returned iff the input argument `tck` is a tuple, otherwise
        a BSpline object is constructed and returned.

    See Also
    --------
    splder, splev, spalde
    BSpline

    Notes
    -----
    The `splder` function is the inverse operation of this function.
    Namely, ``splder(splantider(tck))`` is identical to `tck`, modulo
    rounding error.

    .. versionadded:: 0.13.0

    Examples
    --------
    >>> from scipy.interpolate import splrep, splder, splantider, splev
    >>> import numpy as np
    >>> x = np.linspace(0, np.pi/2, 70)
    >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
    >>> spl = splrep(x, y)

    The derivative is the inverse operation of the antiderivative,
    although some floating point error accumulates:

    >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
    (array(2.1565429877197317), array(2.1565429877201865))

    Antiderivative can be used to evaluate definite integrals:

    >>> ispl = splantider(spl)
    >>> splev(np.pi/2, ispl) - splev(0, ispl)
    2.2572053588768486

    This is indeed an approximation to the complete elliptic integral
    :math:`K(m) = \int_0^{\pi/2} [1 - m\sin^2 x]^{-1/2} dx`:

    >>> from scipy.special import ellipk
    >>> ellipk(0.8)
    2.2572053268208538

    