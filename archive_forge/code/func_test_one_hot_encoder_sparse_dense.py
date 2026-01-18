import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_sparse_dense():
    X = np.array([[3, 2, 1], [0, 1, 1]])
    enc_sparse = OneHotEncoder()
    enc_dense = OneHotEncoder(sparse_output=False)
    X_trans_sparse = enc_sparse.fit_transform(X)
    X_trans_dense = enc_dense.fit_transform(X)
    assert X_trans_sparse.shape == (2, 5)
    assert X_trans_dense.shape == (2, 5)
    assert sparse.issparse(X_trans_sparse)
    assert not sparse.issparse(X_trans_dense)
    assert_array_equal(X_trans_sparse.toarray(), [[0.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
    assert_array_equal(X_trans_sparse.toarray(), X_trans_dense)