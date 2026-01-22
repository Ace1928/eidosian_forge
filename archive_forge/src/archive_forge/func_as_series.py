import cupy
import operator
import warnings
def as_series(alist, trim=True):
    """Returns argument as a list of 1-d arrays.

    Args:
        alist (cupy.ndarray or list of cupy.ndarray): 1-D or 2-D input array.
        trim (bool, optional): trim trailing zeros.

    Returns:
        list of cupy.ndarray: list of 1-D arrays.

    .. seealso:: :func:`numpy.polynomial.polyutils.as_series`

    """
    arrays = []
    for a in alist:
        if a.size == 0:
            raise ValueError('Coefficient array is empty')
        if a.ndim > 1:
            raise ValueError('Coefficient array is not 1-d')
        if a.dtype.kind == 'b':
            raise ValueError('Coefficient arrays have no common type')
        a = a.ravel()
        if trim:
            a = trimseq(a)
        arrays.append(a)
    dtype = cupy.common_type(*arrays)
    ret = [a.astype(dtype, copy=False) for a in arrays]
    return ret