import numpy
import cupy
from cupy import _core
from cupy._logic import content

    Returns ``True`` if all elements are equal or shape consistent,
    i.e., one input array can be broadcasted to create the same
    shape as the other.

    Args:
        a1 (cupy.ndarray): Input array.
        a2 (cupy.ndarray): Input array.

    Returns:
        cupy.ndarray: A boolean 0-dim array.
            ``True`` if equivalent, otherwise ``False``.

    .. seealso:: :func:`numpy.array_equiv`

    