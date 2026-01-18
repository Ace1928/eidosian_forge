import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_python_integer():
    """Check that `OrdinalEncoder` accepts Python integers that are potentially
    larger than 64 bits.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20721
    """
    X = np.array([44253463435747313673, 9867966753463435747313673, 44253462342215747313673, 442534634357764313673]).reshape(-1, 1)
    encoder = OrdinalEncoder().fit(X)
    assert_array_equal(encoder.categories_, np.sort(X, axis=0).T)
    X_trans = encoder.transform(X)
    assert_array_equal(X_trans, [[0], [3], [2], [1]])