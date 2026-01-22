import pytest
import numpy as np
from scipy.optimize import quadratic_assignment, OptimizeWarning
from scipy.optimize._qap import _calc_score as _score
from numpy.testing import assert_equal, assert_, assert_warns
def _range_matrix(a, b):
    mat = np.zeros((a, b))
    for i in range(b):
        mat[:, i] = np.arange(a)
    return mat