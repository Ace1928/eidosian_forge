import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_to_assure_2d_array(self):
    with pytest.raises(ValueError):
        a = array([1, 2, 3])
        b = array([4, 5, 6])
        khatri_rao(a, b)
    with pytest.raises(ValueError):
        a = array([1, 2, 3])
        b = array([[1, 2, 3], [4, 5, 6]])
        khatri_rao(a, b)
    with pytest.raises(ValueError):
        a = array([[1, 2, 3], [7, 8, 9]])
        b = array([4, 5, 6])
        khatri_rao(a, b)