import itertools
import numpy
import cupy
from cupy._core import internal
Rotate an array by 90 degrees in the plane specified by axes.

    Note that ``axes`` argument has been introduced since NumPy v1.12.
    The contents of this document is the same as the original one.

    Args:
        a (~cupy.ndarray): Array of two or more dimensions.
        k (int): Number of times the array is rotated by 90 degrees.
        axes: (tuple of ints): The array is rotated in the plane defined by
            the axes. Axes must be different.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.rot90`

    