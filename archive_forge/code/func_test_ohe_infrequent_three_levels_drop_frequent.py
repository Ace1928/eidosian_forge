import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('drop', ['first', ['b']])
def test_ohe_infrequent_three_levels_drop_frequent(drop):
    """Test three levels and dropping the frequent category."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3]).T
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, max_categories=3, drop=drop).fit(X_train)
    X_test = np.array([['b'], ['c'], ['d']])
    assert_allclose([[0, 0], [1, 0], [0, 1]], ohe.transform(X_test))
    ohe.set_params(handle_unknown='ignore').fit(X_train)
    msg = 'Found unknown categories'
    with pytest.warns(UserWarning, match=msg):
        X_trans = ohe.transform([['b'], ['e']])
    assert_allclose([[0, 0], [0, 0]], X_trans)