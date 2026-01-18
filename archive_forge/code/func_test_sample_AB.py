import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
def test_sample_AB(self):
    A = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
    B = A + 100
    ref = np.array([[[101, 104, 107, 110], [2, 5, 8, 11], [3, 6, 9, 12]], [[1, 4, 7, 10], [102, 105, 108, 111], [3, 6, 9, 12]], [[1, 4, 7, 10], [2, 5, 8, 11], [103, 106, 109, 112]]])
    AB = sample_AB(A=A, B=B)
    assert_allclose(AB, ref)