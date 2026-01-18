import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ohe_infrequent_three_levels_user_cats():
    """Test that the order of the categories provided by a user is respected.
    In this case 'c' is encoded as the first category and 'b' is encoded
    as the second one."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3], dtype=object).T
    ohe = OneHotEncoder(categories=[['c', 'd', 'b', 'a']], sparse_output=False, handle_unknown='infrequent_if_exist', max_categories=3).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [['d', 'a']])
    X_test = [['b'], ['a'], ['c'], ['d'], ['e']]
    expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)
    expected_inv = [['b'], ['infrequent_sklearn'], ['c'], ['infrequent_sklearn'], ['infrequent_sklearn']]
    X_inv = ohe.inverse_transform(X_trans)
    assert_array_equal(expected_inv, X_inv)