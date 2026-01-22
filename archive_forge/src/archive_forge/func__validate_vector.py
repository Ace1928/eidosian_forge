from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
def _validate_vector(self, A, name):
    A = np.atleast_1d(A)
    if A.ndim != 1 or not (np.issubdtype(A.dtype, np.integer) or np.issubdtype(A.dtype, np.floating)):
        message = f'The input `{name}` must be a one-dimensional array of real numbers.'
        raise ValueError(message)
    return A