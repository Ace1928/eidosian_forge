import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy import special
Multidimensional ellipsoid Fourier filter.

    The array is multiplied with the fourier transform of a ellipsoid of
    given sizes.

    Args:
        input (cupy.ndarray): The input array.
        size (float or sequence of float):  The size of the box used for
            filtering. If a float, `size` is the same for all axes. If a
            sequence, `size` has to contain one value for each axis.
        n (int, optional):  If `n` is negative (default), then the input is
            assumed to be the result of a complex fft. If `n` is larger than or
            equal to zero, the input is assumed to be the result of a real fft,
            and `n` gives the length of the array before transformation along
            the real transform direction.
        axis (int, optional): The axis of the real transform (only used when
            ``n > -1``).
        output (cupy.ndarray, optional):
            If given, the result of shifting the input is placed in this array.

    Returns:
        output (cupy.ndarray): The filtered output.
    