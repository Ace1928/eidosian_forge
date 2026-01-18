import re
import sys
from io import StringIO
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('dtype_in, dtype_out', [(np.float32, np.float32), (np.float64, np.float64), (int, np.float64)])
def test_transformer_dtypes_casting(dtype_in, dtype_out):
    X = Xdigits[:100].astype(dtype_in)
    rbm = BernoulliRBM(n_components=16, batch_size=5, n_iter=5, random_state=42)
    Xt = rbm.fit_transform(X)
    assert Xt.dtype == dtype_out, 'transform dtype: {} - original dtype: {}'.format(Xt.dtype, X.dtype)