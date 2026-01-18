import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('sparse_', [False, True])
@pytest.mark.parametrize('X, X_trans', [([[2, 55], [1, 55], [2, 55]], [[0, 1, 1], [0, 0, 0], [0, 1, 1]]), ([['one', 'a'], ['two', 'a'], ['three', 'b'], ['two', 'a']], [[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0]])])
def test_one_hot_encoder_inverse_transform_raise_error_with_unknown(X, X_trans, sparse_):
    """Check that `inverse_transform` raise an error with unknown samples, no
    dropped feature, and `handle_unknow="error`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/14934
    """
    enc = OneHotEncoder(sparse_output=sparse_).fit(X)
    msg = "Samples \\[(\\d )*\\d\\] can not be inverted when drop=None and handle_unknown='error' because they contain all zeros"
    if sparse_:
        X_trans = _convert_container(X_trans, 'sparse')
    with pytest.raises(ValueError, match=msg):
        enc.inverse_transform(X_trans)