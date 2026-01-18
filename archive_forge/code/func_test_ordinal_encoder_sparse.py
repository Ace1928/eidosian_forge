import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_ordinal_encoder_sparse(csr_container):
    """Check that we raise proper error with sparse input in OrdinalEncoder.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19878
    """
    X = np.array([[3, 2, 1], [0, 1, 1]])
    X_sparse = csr_container(X)
    encoder = OrdinalEncoder()
    err_msg = 'Sparse data was passed, but dense data is required'
    with pytest.raises(TypeError, match=err_msg):
        encoder.fit(X_sparse)
    with pytest.raises(TypeError, match=err_msg):
        encoder.fit_transform(X_sparse)
    X_trans = encoder.fit_transform(X)
    X_trans_sparse = csr_container(X_trans)
    with pytest.raises(TypeError, match=err_msg):
        encoder.inverse_transform(X_trans_sparse)