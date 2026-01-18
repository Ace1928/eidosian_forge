import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('handle_unknown', ['ignore', 'infrequent_if_exist'])
def test_ohe_drop_first_handle_unknown_ignore_warns(handle_unknown):
    """Check drop='first' and handle_unknown='ignore'/'infrequent_if_exist'
    during transform."""
    X = [['a', 0], ['b', 2], ['b', 1]]
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown=handle_unknown)
    X_trans = ohe.fit_transform(X)
    X_expected = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0]])
    assert_allclose(X_trans, X_expected)
    X_test = [['c', 3]]
    X_expected = np.array([[0, 0, 0]])
    warn_msg = 'Found unknown categories in columns \\[0, 1\\] during transform. These unknown categories will be encoded as all zeros'
    with pytest.warns(UserWarning, match=warn_msg):
        X_trans = ohe.transform(X_test)
    assert_allclose(X_trans, X_expected)
    X_inv = ohe.inverse_transform(X_expected)
    assert_array_equal(X_inv, np.array([['a', 0]], dtype=object))