def _np_transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    axes : list of ints, optional
        By default, reverse the dimensions,
        otherwise permute the axes according to the values given.

    Returns
    -------
    p : ndarray
        a with its axes permuted.

    Notes
    -----
    This function differs from the original `numpy.transpose
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html>`_ in
    the following way(s):

    - only ndarray is accepted as valid input, python iterables are not supported
    - the operator always returns an `ndarray` that does not share the memory with the input

    Examples
    --------
    >>> x = np.arange(4).reshape((2,2))
    >>> x
    array([[0., 1.],
           [2., 3.]])
    >>> np.transpose(x)
    array([[0., 2.],
           [1., 3.]])
    >>> x = np.ones((1, 2, 3))
    >>> np.transpose(x, (1, 0, 2)).shape
    (2, 1, 3)
    """
    pass