def _npx_reshape(a, newshape, reverse=False, order='C'):
    """
    Gives a new shape to an array without changing its data.
    This function always returns a copy of the input array if
    ``out`` is not provided.

    Parameters
    ----------
    a : ndarray
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape.
        If an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is inferred
        from the length of the array and remaining dimensions.
        -2 to -6 are used for data manipulation.

        - -2 copy this dimension from the input to the output shape.
        - -3 will skip current dimension if and only if the current dim size is one.
        - -4 copy all remain of the input dimensions to the output shape.
        - -5 use the product of two consecutive dimensions of the input
          shape as the output.
        - -6 split one dimension of the input into two dimensions passed
          subsequent to -6 in the new shape.

    reverse : bool, optional
        If set to true, the special values will be inferred from right to left.
    order : {'C'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. Other order types such as 'F'/'A'
        may be added in the future.

    Returns
    -------
    reshaped_array : ndarray
        It will be always a copy of the original array. This behavior is different
        from the official NumPy ``reshape`` operator where views of the original array may be
        generated.

    Examples
    --------
    >>> x = np.ones((2, 3, 8))
    >>> npx.reshape(x, (-2, -2, 2, -1)).shape
    (2, 3, 2, 4)
    >>> x = np.ones((8, 3, 3, 3, 4, 4))
    >>> npx.reshape(x, (-6, 2, -1, -4)).shape
    (2, 4, 3, 3, 3, 4, 4)
    >>> x = np.ones((8, 3, 3, 3, 4, 4))
    >>> npx.reshape(x, (-5, -4)).shape
    (24, 3, 3, 4, 4)
    >>> x = np.ones((8, 1, 1, 1, 3))
    >>> npx.reshape(x, (-2, -3, -3, -3, -2)).shape
    (8, 3)
    >>> x = np.ones((8, 3, 3, 3, 3, 8))
    >>> npx.reshape(x, (-4, -5), reverse=True).shape
    (8, 3, 3, 3, 24)
    >>> x = np.ones((8, 3, 2, 4, 8))
    >>> npx.reshape(x, (-4, -1, 2, -6), reverse=True).shape
    (8, 3, 2, 4, 4, 2)
    """
    pass