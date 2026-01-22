def _np_amin(a, axis=None, keepdims=False, out=None):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : int, optional
        Axis along which to operate.  By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    amin : ndarray
        Minimum of `a`. If `axis` is None, the result is an array of dimension 1.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    max :
        The maximum value of an array along a given axis, ignoring any nan.
    minimum :
        Element-wise minimum of two arrays, ignoring any nan.

    Notes
    -----
    NaN in the orginal `numpy` is denoted as nan and will be ignored.

    Don't use `amin` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``amin(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0., 1.],
        [2., 3.]])
    >>> np.amin(a)           # Minimum of the flattened array
    array(0.)
    >>> np.amin(a, axis=0)   # Minima along the first axis
    array([0., 1.])
    >>> np.amin(a, axis=1)   # Minima along the second axis
    array([0., 2.])
    >>> b = np.arange(5, dtype=np.float32)
    >>> b[2] = np.nan
    >>> np.amin(b)
    array(0.) # nan will be ignored
    """
    pass