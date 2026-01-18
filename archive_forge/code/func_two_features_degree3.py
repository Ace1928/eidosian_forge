import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.fixture()
def two_features_degree3():
    X = np.arange(6).reshape((3, 2))
    x1 = X[:, :1]
    x2 = X[:, 1:]
    P = np.hstack([x1 ** 0 * x2 ** 0, x1 ** 1 * x2 ** 0, x1 ** 0 * x2 ** 1, x1 ** 2 * x2 ** 0, x1 ** 1 * x2 ** 1, x1 ** 0 * x2 ** 2, x1 ** 3 * x2 ** 0, x1 ** 2 * x2 ** 1, x1 ** 1 * x2 ** 2, x1 ** 0 * x2 ** 3])
    return (X, P)