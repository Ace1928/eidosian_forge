from cupy import _core
from cupy._core import fusion
from cupy._math import ufunc
Rounds to the given number of decimals.

    Args:
        a (cupy.ndarray): The source array.
        decimals (int): Number of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of positions to
            the left of the decimal point.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: Rounded array.

    .. seealso:: :func:`numpy.around`

    