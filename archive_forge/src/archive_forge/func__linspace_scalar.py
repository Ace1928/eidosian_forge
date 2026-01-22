import math
import numpy
import cupy
from cupy import _core
def _linspace_scalar(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    """Returns an array with evenly-spaced values within a given interval.

    Instead of specifying the step width like :func:`cupy.arange`, this
    function requires the total number of elements specified.

    Args:
        start: Start of the interval.
        stop: End of the interval.
        num: Number of elements.
        endpoint (bool): If ``True``, the stop value is included as the last
            element. Otherwise, the stop value is omitted.
        retstep (bool): If ``True``, this function returns (array, step).
            Otherwise, it returns only the array.
        dtype: Data type specifier. It is inferred from the start and stop
            arguments by default.

    Returns:
        cupy.ndarray: The 1-D array of ranged values.

    """
    dt = cupy.result_type(start, stop, float(num))
    if dtype is None:
        dtype = dt
    ret = cupy.empty((num,), dtype=dt)
    div = num - 1 if endpoint else num
    if div <= 0:
        if num > 0:
            ret.fill(start)
        step = float('nan')
    else:
        step = float(stop - start) / div
        stop = float(stop)
        if step == 0.0:
            _linspace_ufunc_underflow(start, stop - start, div, ret)
        else:
            _linspace_ufunc(start, step, ret)
        if endpoint:
            ret[-1] = stop
    if cupy.issubdtype(dtype, cupy.integer):
        cupy.floor(ret, out=ret)
    ret = ret.astype(dtype, copy=False)
    if retstep:
        return (ret, step)
    else:
        return ret