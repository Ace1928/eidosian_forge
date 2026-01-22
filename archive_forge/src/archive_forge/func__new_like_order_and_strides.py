import numpy
import cupy
from cupy._core.internal import _get_strides_for_order_K, _update_order_char
def _new_like_order_and_strides(a, dtype, order, shape=None, *, get_memptr=True):
    """
    Determine order and strides as in NumPy's PyArray_NewLikeArray.

    (see: numpy/core/src/multiarray/ctors.c)
    """
    order = order.upper()
    if order not in ['C', 'F', 'K', 'A']:
        raise ValueError('order not understood: {}'.format(order))
    if numpy.isscalar(shape):
        shape = (shape,)
    if order == 'K' and shape is not None and (len(shape) != a.ndim):
        return ('C', None, None)
    order = chr(_update_order_char(a.flags.c_contiguous, a.flags.f_contiguous, ord(order)))
    if order == 'K':
        strides = _get_strides_for_order_K(a, numpy.dtype(dtype), shape)
        order = 'C'
        memptr = cupy.empty(a.size, dtype=dtype).data if get_memptr else None
        return (order, strides, memptr)
    else:
        return (order, None, None)