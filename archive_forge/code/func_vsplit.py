import numpy
from cupy import _core
def vsplit(ary, indices_or_sections):
    """Splits an array into multiple sub arrays along the first axis.

    This is equivalent to ``split`` with ``axis=0``.

    .. seealso:: :func:`cupy.split` for more detail, :func:`numpy.dsplit`

    """
    if ary.ndim <= 1:
        raise ValueError('Cannot vsplit an array with less than 2 dimensions')
    return split(ary, indices_or_sections, 0)