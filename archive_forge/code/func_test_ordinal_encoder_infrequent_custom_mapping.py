import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_infrequent_custom_mapping():
    """Check behavior of unknown_value and encoded_missing_value with infrequent."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3 + [np.nan]], dtype=object).T
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=2, max_categories=2, encoded_missing_value=3).fit(X_train)
    assert_array_equal(ordinal.infrequent_categories_, [['a', 'c', 'd']])
    X_test = np.array([['a'], ['b'], ['c'], ['d'], ['e'], [np.nan]], dtype=object)
    expected_trans = [[1], [0], [1], [1], [2], [3]]
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)