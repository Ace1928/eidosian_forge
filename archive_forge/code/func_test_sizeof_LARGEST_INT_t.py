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
def test_sizeof_LARGEST_INT_t():
    if sys.platform == 'win32' or (sys.maxsize <= 2 ** 32 and sys.platform != 'emscripten'):
        expected_size = 8
    else:
        expected_size = 16
    assert _get_sizeof_LARGEST_INT_t() == expected_size