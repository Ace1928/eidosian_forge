def _np_atleast_2d(*arys):
    """
    Convert inputs to arrays with at least two dimensions.

    Parameters
    ----------
    arys1, arys2, ... : ndarray
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or list of arrays, each with a.ndim >= 2. Copies are made only if necessary.

    See also
    --------
    atleast_1d, atleast_3d

    Examples
    --------
    >>> np.atleast_2d(3.0)
    array([[3.]])
    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array([[0., 1., 2.]])
    >>> np.atleast_2d(np.array(1), np.array([1, 2]), np.array([[1, 2]]))
    [array([[1.]]), array([[1., 2.]]), array([[1., 2.]])]
    """
    pass