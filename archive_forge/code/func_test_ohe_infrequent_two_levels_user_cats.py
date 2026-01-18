import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ohe_infrequent_two_levels_user_cats():
    """Test that the order of the categories provided by a user is respected."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3], dtype=object).T
    ohe = OneHotEncoder(categories=[['c', 'd', 'a', 'b']], sparse_output=False, handle_unknown='infrequent_if_exist', max_categories=2).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [['c', 'd', 'a']])
    X_test = [['b'], ['a'], ['c'], ['d'], ['e']]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)
    expected_inv = [[col] for col in ['b'] + ['infrequent_sklearn'] * 4]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)