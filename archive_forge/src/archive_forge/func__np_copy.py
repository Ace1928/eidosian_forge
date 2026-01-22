def _np_copy(a, out=None):
    """
    Return an array copy of the given object.

    Parameters
    ----------
    a : ndarray
        Input data.
    out : ndarray or None, optional
        Alternative output array in which to place the result. It must have
        the same shape and dtype as the expected output.

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    Notes
    -------
    This function differs from the original `numpy.copy
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.copy.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - ``out`` param: cannot perform auto broadcasting. ``out`` ndarray's shape must be the same as the expected output.
    - ``out`` param: cannot perform auto type cast. ``out`` ndarray's dtype must be the same as the expected output.
    - Does not support "order" parameter.

    Examples
    --------
    Create an array x, with a reference y and a copy z:

    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when ``x`` is modified, ``y`` is also modified, but not ``z``:

    >>> x[0] = 10
    >>> x[0] == y[0]
    array([1.])
    >>> x[0] == z[0]
    array([0.])
    """
    pass