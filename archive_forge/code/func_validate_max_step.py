from itertools import groupby
from warnings import warn
import numpy as np
from scipy.sparse import find, coo_matrix
def validate_max_step(max_step):
    """Assert that max_Step is valid and return it."""
    if max_step <= 0:
        raise ValueError('`max_step` must be positive.')
    return max_step