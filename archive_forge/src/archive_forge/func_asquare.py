from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def asquare(cdfvals, axis=0):
    """vectorized Anderson Darling A^2, Stephens 1974"""
    ndim = len(cdfvals.shape)
    nobs = cdfvals.shape[axis]
    slice_reverse = [slice(None)] * ndim
    islice = [None] * ndim
    islice[axis] = slice(None)
    slice_reverse[axis] = slice(None, None, -1)
    asqu = -((2.0 * np.arange(1.0, nobs + 1)[tuple(islice)] - 1) * (np.log(cdfvals) + np.log(1 - cdfvals[tuple(slice_reverse)])) / nobs).sum(axis) - nobs
    return asqu