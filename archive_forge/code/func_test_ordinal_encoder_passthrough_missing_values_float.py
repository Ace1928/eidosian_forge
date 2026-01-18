import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('encoded_missing_value', [np.nan, -2])
def test_ordinal_encoder_passthrough_missing_values_float(encoded_missing_value):
    """Test ordinal encoder with nan on float dtypes."""
    X = np.array([[np.nan, 3.0, 1.0, 3.0]], dtype=np.float64).T
    oe = OrdinalEncoder(encoded_missing_value=encoded_missing_value).fit(X)
    assert len(oe.categories_) == 1
    assert_allclose(oe.categories_[0], [1.0, 3.0, np.nan])
    X_trans = oe.transform(X)
    assert_allclose(X_trans, [[encoded_missing_value], [1.0], [0.0], [1.0]])
    X_inverse = oe.inverse_transform(X_trans)
    assert_allclose(X_inverse, X)