import numpy
from cupy import cuda
from cupy._creation.basic import _new_like_order_and_strides
from cupy._core import internal
def zeros_pinned(shape, dtype=float, order='C'):
    """Returns a new, zero-initialized NumPy array with the given shape
    and dtype.

    This is a convenience function which is just :func:`numpy.zeros`,
    except that the underlying memory is pinned/pagelocked.

    Args:
        shape (int or tuple of ints): Dimensionalities of the array.
        dtype: Data type specifier.
        order ({'C', 'F'}): Row-major (C-style) or column-major
            (Fortran-style) order.

    Returns:
        numpy.ndarray: An array filled with zeros.

    .. seealso:: :func:`numpy.zeros`

    """
    out = empty_pinned(shape, dtype, order)
    numpy.copyto(out, 0, casting='unsafe')
    return out