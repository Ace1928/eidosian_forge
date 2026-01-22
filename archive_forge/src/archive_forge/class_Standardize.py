from functools import wraps
import numpy as np
from patsy.util import (atleast_2d_column_default,
class Standardize(object):
    """standardize(x, center=True, rescale=True, ddof=0)

    A stateful transform that standardizes input data, i.e. it subtracts the
    mean and divides by the sample standard deviation.

    Either centering or rescaling or both can be disabled by use of keyword
    arguments. The `ddof` argument controls the delta degrees of freedom when
    computing the standard deviation (cf. :func:`numpy.std`). The default of
    ``ddof=0`` produces the maximum likelihood estimate; use ``ddof=1`` if you
    prefer the square root of the unbiased estimate of the variance.

    If input has multiple columns, standardizes each column separately.

    .. note:: This function computes the mean and standard deviation using a
       memory-efficient online algorithm, making it suitable for use with
       large incrementally processed data-sets.
    """

    def __init__(self):
        self.current_n = 0
        self.current_mean = None
        self.current_M2 = None

    def memorize_chunk(self, x, center=True, rescale=True, ddof=0):
        x = atleast_2d_column_default(x)
        if self.current_mean is None:
            self.current_mean = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
            self.current_M2 = np.zeros(x.shape[1], dtype=wide_dtype_for(x))
        for i in range(x.shape[0]):
            self.current_n += 1
            delta = x[i, :] - self.current_mean
            self.current_mean += delta / self.current_n
            self.current_M2 += delta * (x[i, :] - self.current_mean)

    def memorize_finish(self):
        pass

    def transform(self, x, center=True, rescale=True, ddof=0):
        x = asarray_or_pandas(x, copy=True, dtype=float)
        x_2d = atleast_2d_column_default(x, preserve_pandas=True)
        if center:
            x_2d -= self.current_mean
        if rescale:
            x_2d /= np.sqrt(self.current_M2 / (self.current_n - ddof))
        return pandas_friendly_reshape(x_2d, x.shape)
    __getstate__ = no_pickling