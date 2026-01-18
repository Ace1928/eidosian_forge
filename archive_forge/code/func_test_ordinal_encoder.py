import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X', [[['abc', 2, 55], ['def', 1, 55]], np.array([[10, 2, 55], [20, 1, 55]]), np.array([['a', 'B', 'cat'], ['b', 'A', 'cat']], dtype=object)], ids=['mixed', 'numeric', 'object'])
def test_ordinal_encoder(X):
    enc = OrdinalEncoder()
    exp = np.array([[0, 1, 0], [1, 0, 0]], dtype='int64')
    assert_array_equal(enc.fit_transform(X), exp.astype('float64'))
    enc = OrdinalEncoder(dtype='int64')
    assert_array_equal(enc.fit_transform(X), exp)