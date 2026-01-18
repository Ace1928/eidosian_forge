import cupy
import operator
import warnings
def trimcoef(c, tol=0):
    """Removes small trailing coefficients from a polynomial.

    Args:
        c(cupy.ndarray): 1d array of coefficients from lowest to highest order.
        tol(number, optional): trailing coefficients whose absolute value are
            less than or equal to ``tol`` are trimmed.

    Returns:
        cupy.ndarray: trimmed 1d array.

    .. seealso:: :func:`numpy.polynomial.polyutils.trimcoef`

    """
    if tol < 0:
        raise ValueError('tol must be non-negative')
    if c.size == 0:
        raise ValueError('Coefficient array is empty')
    if c.ndim > 1:
        raise ValueError('Coefficient array is not 1-d')
    if c.dtype.kind == 'b':
        raise ValueError('bool inputs are not allowed')
    if c.ndim == 0:
        c = c.ravel()
    c = c.astype(cupy.common_type(c), copy=False)
    filt = (cupy.abs(c) > tol)[::-1]
    ind = c.size - cupy._manipulation.add_remove._first_nonzero_krnl(filt, c.size).item()
    if ind == 0:
        return cupy.zeros_like(c[:1])
    return c[:ind]