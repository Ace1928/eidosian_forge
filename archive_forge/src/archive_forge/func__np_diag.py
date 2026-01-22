def _np_diag(array, k=0):
    """
    Extracts a diagonal or constructs a diagonal array.
    - 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero.
    - 2-D arrays: extracts the k-th Diagonal

    Parameters
    ----------
    array : ndarray
        The array to apply diag method.
    k : offset
        extracts or constructs kth diagonal given input array

    Examples
    --------
    >>> x = np.arange(9).reshape((3,3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])
    """
    pass