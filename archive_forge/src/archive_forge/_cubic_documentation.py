import cupy
from cupyx.scipy.interpolate._interpolate import PPoly

    Akima interpolator

    Fit piecewise cubic polynomials, given vectors x and y. The interpolation
    method by Akima uses a continuously differentiable sub-spline built from
    piecewise cubic polynomials. The resultant curve passes through the given
    data points and will appear smooth and natural [1]_.

    Parameters
    ----------
    x : ndarray, shape (m, )
        1-D array of monotonically increasing real values.
    y : ndarray, shape (m, ...)
        N-D array of real values. The length of ``y`` along the first axis
        must be equal to the length of ``x``.
    axis : int, optional
        Specifies the axis of ``y`` along which to interpolate. Interpolation
        defaults to the first axis of ``y``.

    See Also
    --------
    CubicHermiteSpline : Piecewise-cubic interpolator.
    PchipInterpolator : PCHIP 1-D monotonic cubic interpolator.
    PPoly : Piecewise polynomial in terms of coefficients and breakpoints

    Notes
    -----
    Use only for precise data, as the fitted curve passes through the given
    points exactly. This routine is useful for plotting a pleasingly smooth
    curve through a few given points for purposes of plotting.

    References
    ----------
    .. [1] A new method of interpolation and smooth curve fitting based
        on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4),
        589-602.
    