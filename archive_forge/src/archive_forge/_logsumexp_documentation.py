import cupy as cp
Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : cupy.ndarray
        Input array
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default
        `axis` is None, and all elements are summed
    keepdims : bool, optional
        If this is set to True, the axes which are reduced
        are left in the result as dimensions with size one. With
        this option, the result will broadcast correctly
        against the original array
    b : cupy.ndarray, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False

    Returns
    -------
    res : cupy.ndarray
        The result, ``cp.log(cp.sum(cp.exp(a)))`` calculated
        in a numerically more stable way. If `b` is given then
        ``cp.log(cp.sum(b*cp.exp(a)))`` is returned
    sgn : cupy.ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign of
        the result. If False, only onw result is returned

    See Also
    --------
    scipy.special.logsumexp

    