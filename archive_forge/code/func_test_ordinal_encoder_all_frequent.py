import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kwargs', [{'max_categories': 6}, {'min_frequency': 2}])
def test_ordinal_encoder_all_frequent(kwargs):
    """All categories are considered frequent have same encoding as default encoder."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3], dtype=object).T
    adjusted_encoder = OrdinalEncoder(**kwargs, handle_unknown='use_encoded_value', unknown_value=-1).fit(X_train)
    default_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(X_train)
    X_test = [['a'], ['b'], ['c'], ['d'], ['e']]
    assert_allclose(adjusted_encoder.transform(X_test), default_encoder.transform(X_test))