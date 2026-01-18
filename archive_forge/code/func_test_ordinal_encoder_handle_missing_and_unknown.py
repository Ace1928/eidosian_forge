import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X, expected_X_trans, X_test', [(np.array([[1.0, np.nan, 3.0]]).T, np.array([[0.0, np.nan, 1.0]]).T, np.array([[4.0]])), (np.array([[1.0, 4.0, 3.0]]).T, np.array([[0.0, 2.0, 1.0]]).T, np.array([[np.nan]])), (np.array([['c', np.nan, 'b']], dtype=object).T, np.array([[1.0, np.nan, 0.0]]).T, np.array([['d']], dtype=object)), (np.array([['c', 'a', 'b']], dtype=object).T, np.array([[2.0, 0.0, 1.0]]).T, np.array([[np.nan]], dtype=object))])
def test_ordinal_encoder_handle_missing_and_unknown(X, expected_X_trans, X_test):
    """Test the interaction between missing values and handle_unknown"""
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_trans = oe.fit_transform(X)
    assert_allclose(X_trans, expected_X_trans)
    assert_allclose(oe.transform(X_test), [[-1.0]])