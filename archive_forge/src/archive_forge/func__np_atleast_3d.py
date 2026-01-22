def _np_atleast_3d(*arys):
    """
    Convert inputs to arrays with at least three dimension.

    Parameters
    ----------
    arys1, arys2, ... : ndarray
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or list of arrays, each with a.ndim >= 3.
        For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1),
        and a 2-D array of shape (M, N) becomes a view of shape (M, N, 1).

    See also
    --------
    atleast_1d, atleast_2d

    Examples
    --------
    >>> np.atleast_3d(3.0)
    array([[[3.]]])
    >>> x = np.arange(3.0)
    >>> np.atleast_3d(x).shape
    (1, 3, 1)
    >>> x = np.arange(12.0).reshape(4,3)
    >>> np.atleast_3d(x).shape
    (4, 3, 1)
    >>> for arr in np.atleast_3d(np.array([1, 2]), np.array([[1, 2]]), np.array([[[1, 2]]])):
    ...     print(arr, arr.shape)
    ...
    [[[1.]
      [2.]]] (1, 2, 1)
    [[[1.]
      [2.]]] (1, 2, 1)
    [[[1. 2.]]] (1, 1, 2)
    """
    pass