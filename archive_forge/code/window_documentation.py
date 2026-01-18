import numpy
import cupy
from cupy import _core
Return the Kaiser window.
    The Kaiser window is a taper formed by using a Bessel function.

    .. math::  w(n) = I_0\left( \beta \sqrt{1-\frac{4n^2}{(M-1)^2}}
               \right)/I_0(\beta)

    with

    .. math:: \quad -\frac{M-1}{2} \leq n \leq \frac{M-1}{2}

    where :math:`I_0` is the modified zeroth-order Bessel function.

     Args:
        M (int):
            Number of points in the output window. If zero or less, an empty
            array is returned.
        beta (float):
            Shape parameter for window

    Returns:
        ~cupy.ndarray:  The window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd).

    .. seealso:: :func:`numpy.kaiser`
    