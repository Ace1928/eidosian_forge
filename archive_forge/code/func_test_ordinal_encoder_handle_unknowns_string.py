import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_handle_unknowns_string():
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-2)
    X_fit = np.array([['a', 'x'], ['b', 'y'], ['c', 'z']], dtype=object)
    X_trans = np.array([['c', 'xy'], ['bla', 'y'], ['a', 'x']], dtype=object)
    enc.fit(X_fit)
    X_trans_enc = enc.transform(X_trans)
    exp = np.array([[2, -2], [-2, 1], [0, 0]], dtype='int64')
    assert_array_equal(X_trans_enc, exp)
    X_trans_inv = enc.inverse_transform(X_trans_enc)
    inv_exp = np.array([['c', None], [None, 'y'], ['a', 'x']], dtype=object)
    assert_array_equal(X_trans_inv, inv_exp)