import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('output_dtype', [np.int32, np.float32, np.float64])
@pytest.mark.parametrize('input_dtype', [np.int32, np.float32, np.float64])
def test_one_hot_encoder_dtype(input_dtype, output_dtype):
    X = np.asarray([[0, 1]], dtype=input_dtype).T
    X_expected = np.asarray([[1, 0], [0, 1]], dtype=output_dtype)
    oh = OneHotEncoder(categories='auto', dtype=output_dtype)
    assert_array_equal(oh.fit_transform(X).toarray(), X_expected)
    assert_array_equal(oh.fit(X).transform(X).toarray(), X_expected)
    oh = OneHotEncoder(categories='auto', dtype=output_dtype, sparse_output=False)
    assert_array_equal(oh.fit_transform(X), X_expected)
    assert_array_equal(oh.fit(X).transform(X), X_expected)