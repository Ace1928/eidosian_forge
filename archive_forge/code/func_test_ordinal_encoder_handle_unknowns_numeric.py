import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('dtype', [float, int])
def test_ordinal_encoder_handle_unknowns_numeric(dtype):
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-999)
    X_fit = np.array([[1, 7], [2, 8], [3, 9]], dtype=dtype)
    X_trans = np.array([[3, 12], [23, 8], [1, 7]], dtype=dtype)
    enc.fit(X_fit)
    X_trans_enc = enc.transform(X_trans)
    exp = np.array([[2, -999], [-999, 1], [0, 0]], dtype='int64')
    assert_array_equal(X_trans_enc, exp)
    X_trans_inv = enc.inverse_transform(X_trans_enc)
    inv_exp = np.array([[3, None], [None, 8], [1, 7]], dtype=object)
    assert_array_equal(X_trans_inv, inv_exp)