import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_infrequent_three_levels_user_cats():
    """Test that the order of the categories provided by a user is respected.

    In this case 'c' is encoded as the first category and 'b' is encoded
    as the second one.
    """
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3], dtype=object).T
    ordinal = OrdinalEncoder(categories=[['c', 'd', 'b', 'a']], max_categories=3, handle_unknown='use_encoded_value', unknown_value=-1).fit(X_train)
    assert_array_equal(ordinal.categories_, [['c', 'd', 'b', 'a']])
    assert_array_equal(ordinal.infrequent_categories_, [['d', 'a']])
    X_test = [['a'], ['b'], ['c'], ['d'], ['z']]
    expected_trans = [[2], [1], [0], [2], [-1]]
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)
    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = [['infrequent_sklearn'], ['b'], ['c'], ['infrequent_sklearn'], [None]]
    assert_array_equal(X_inverse, expected_inverse)