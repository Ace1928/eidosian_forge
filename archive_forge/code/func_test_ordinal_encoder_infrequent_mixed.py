import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_infrequent_mixed():
    """Test when feature 0 has infrequent categories and feature 1 does not."""
    X = np.column_stack(([0, 1, 3, 3, 3, 3, 2, 0, 3], [0, 0, 0, 0, 1, 1, 1, 1, 1]))
    ordinal = OrdinalEncoder(max_categories=3).fit(X)
    assert_array_equal(ordinal.infrequent_categories_[0], [1, 2])
    assert ordinal.infrequent_categories_[1] is None
    X_test = [[3, 0], [1, 1]]
    expected_trans = [[1, 0], [2, 1]]
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)
    X_inverse = ordinal.inverse_transform(X_trans)
    expected_inverse = np.array([[3, 0], ['infrequent_sklearn', 1]], dtype=object)
    assert_array_equal(X_inverse, expected_inverse)