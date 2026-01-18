import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ohe_infrequent_handle_unknown_error():
    """Test that different parameters for combining 'a', and 'd' into
    the infrequent category works as expected."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3]).T
    ohe = OneHotEncoder(handle_unknown='error', sparse_output=False, max_categories=3).fit(X_train)
    assert_array_equal(ohe.infrequent_categories_, [['a', 'd']])
    X_test = [['b'], ['a'], ['c'], ['d']]
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
    X_trans = ohe.transform(X_test)
    assert_allclose(expected, X_trans)
    X_test = [['bad']]
    msg = "Found unknown categories \\['bad'\\] in column 0"
    with pytest.raises(ValueError, match=msg):
        ohe.transform(X_test)