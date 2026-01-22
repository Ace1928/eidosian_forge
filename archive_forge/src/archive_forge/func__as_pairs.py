import numbers
import numpy
import cupy
def _as_pairs(x, ndim, as_index=False):
    """Broadcasts `x` to an array with shape (`ndim`, 2).

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.

    Args:
      x(scalar or array-like, optional): The object to broadcast to the shape
          (`ndim`, 2).
      ndim(int): Number of pairs the broadcasted `x` will have.
      as_index(bool, optional): If `x` is not None, try to round each
          element of `x` to an integer (dtype `cupy.intp`) and ensure every
          element is positive. (Default value = False)

    Returns:
      nested iterables, shape (`ndim`, 2): The broadcasted version of `x`.
    """
    if x is None:
        return ((None, None),) * ndim
    elif isinstance(x, numbers.Number):
        if as_index:
            x = round(x)
        return ((x, x),) * ndim
    x = numpy.array(x)
    if as_index:
        x = numpy.asarray(numpy.round(x), dtype=numpy.intp)
    if x.ndim < 3:
        if x.size == 1:
            x = x.ravel()
            if as_index and x < 0:
                raise ValueError("index can't contain negative values")
            return ((x[0], x[0]),) * ndim
        if x.size == 2 and x.shape != (2, 1):
            x = x.ravel()
            if as_index and (x[0] < 0 or x[1] < 0):
                raise ValueError("index can't contain negative values")
            return ((x[0], x[1]),) * ndim
    if as_index and x.min() < 0:
        raise ValueError("index can't contain negative values")
    x_view = x.view()
    x_view.shape = (ndim, 2)
    return x_view.tolist()