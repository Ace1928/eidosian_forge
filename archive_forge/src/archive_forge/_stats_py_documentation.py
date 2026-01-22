import cupy
Calculate the relative z-scores.

    Return an array of z-scores, i.e., scores that are standardized
    to zero mean and unit variance, where mean and variance are
    calculated from the comparison array.

    Parameters
    ----------
    scores : array-like
        The input for which z-scores are calculated
    compare : array-like
        The input from which the mean and standard deviation of
        the normalization are taken; assumed to have the same
        dimension as `scores`
    axis : int or None, optional
        Axis over which mean and variance of `compare` are calculated.
        Default is 0. If None, compute over the whole array `scores`
    ddof : int, optional
        Degrees of freedom correction in the calculation of the
        standard deviation. Default is 0
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle the occurrence of nans in `compare`.
        'propagate' returns nan, 'raise' raises an exception, 'omit'
        performs the calculations ignoring nan values. Default is
        'propagate'. Note that when the value is 'omit', nans in `scores`
        also propagate to the output, but they do not affect the z-scores
        computed for the non-nan values

    Returns
    -------
    zscore : array-like
        Z-scores, in the same shape as `scores`

    