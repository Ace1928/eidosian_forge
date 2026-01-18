import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_drop_equals_if_binary():
    X = [[10, 'yes'], [20, 'no'], [30, 'yes']]
    expected = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    expected_drop_idx = np.array([None, 0])
    ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
    result = ohe.fit_transform(X)
    assert_array_equal(ohe.drop_idx_, expected_drop_idx)
    assert_allclose(result, expected)
    X = [['true', 'a'], ['false', 'a'], ['false', 'a']]
    expected = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    expected_drop_idx = np.array([0, None])
    ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
    result = ohe.fit_transform(X)
    assert_array_equal(ohe.drop_idx_, expected_drop_idx)
    assert_allclose(result, expected)