import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('drop', ['if_binary', 'first', ['b']])
def test_ohe_infrequent_two_levels_drop_frequent(drop):
    """Test two levels and dropping the frequent category."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3]).T
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, max_categories=2, drop=drop).fit(X_train)
    assert ohe.categories_[0][ohe.drop_idx_[0]] == 'b'
    X_test = np.array([['b'], ['c']])
    X_trans = ohe.transform(X_test)
    assert_allclose([[0], [1]], X_trans)
    feature_names = ohe.get_feature_names_out()
    assert_array_equal(['x0_infrequent_sklearn'], feature_names)
    X_inverse = ohe.inverse_transform(X_trans)
    assert_array_equal([['b'], ['infrequent_sklearn']], X_inverse)