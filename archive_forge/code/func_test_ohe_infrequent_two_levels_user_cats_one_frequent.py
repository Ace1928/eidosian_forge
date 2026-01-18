import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kwargs', [{'max_categories': 3, 'min_frequency': 1}, {'min_frequency': 4}])
def test_ohe_infrequent_two_levels_user_cats_one_frequent(kwargs):
    """'a' is the only frequent category, all other categories are infrequent."""
    X_train = np.array([['a'] * 5 + ['e'] * 30], dtype=object).T
    ohe = OneHotEncoder(categories=[['c', 'd', 'a', 'b']], sparse_output=False, handle_unknown='infrequent_if_exist', **kwargs).fit(X_train)
    X_test = [['a'], ['b'], ['c'], ['d'], ['e']]
    expected = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)
    drops = ['first', 'if_binary', ['a']]
    X_test = [['a'], ['c']]
    for drop in drops:
        ohe.set_params(drop=drop).fit(X_train)
        assert_allclose([[0], [1]], ohe.transform(X_test))