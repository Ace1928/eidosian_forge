import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def validate_vector_shape(name, shape, nrows, nobs):
    """
    Validate the shape of a possibly time-varying vector, or raise an exception

    Parameters
    ----------
    name : str
        The name of the vector being validated (used in exception messages)
    shape : array_like
        The shape of the vector to be validated. May be of size 1 or (if
        the vector is time-varying) 2.
    nrows : int
        The expected number of rows (elements of the vector).
    nobs : int
        The number of observations (used to validate the last dimension of a
        time-varying vector)

    Raises
    ------
    ValueError
        If the vector is not of the desired shape.
    """
    ndim = len(shape)
    if ndim not in [1, 2]:
        raise ValueError('Invalid value for %s vector. Requires a 1- or 2-dimensional array, got %d dimensions' % (name, ndim))
    if not shape[0] == nrows:
        raise ValueError('Invalid dimensions for %s vector: requires %d rows, got %d' % (name, nrows, shape[0]))
    if nobs is None and (not (ndim == 1 or shape[-1] == 1)):
        raise ValueError('Invalid dimensions for %s vector: time-varying vectors cannot be given unless `nobs` is specified (implicitly when a dataset is bound or else set explicity)' % name)
    if ndim == 2 and (not shape[1] in [1, nobs]):
        raise ValueError('Invalid dimensions for time-varying %s vector. Requires shape (*,%d), got %s' % (name, nobs, str(shape)))