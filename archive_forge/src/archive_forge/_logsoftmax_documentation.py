import cupy as cp
Compute logarithm of softmax function

    Parameters
    ----------
    x : array-like
        Input array
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax
        will be  computed over the entire array `x`

    Returns
    -------
    s : cupy.ndarry
        An array with the same shape as `x`. Exponential of the
        result will sum to 1 along the specified axis. If `x` is a
        scalar, a scalar is returned

    