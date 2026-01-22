from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
def _dot_diag(x, d):
    return x * d if x.ndim < 2 else x * np.expand_dims(d, -2)