import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_missing_appears_infrequent():
    """Check behavior when missing value appears infrequently."""
    X = np.array([[np.nan] + ['dog'] * 10 + ['cat'] * 5 + ['snake'] + ['deer'], ['red'] * 9 + ['green'] * 9], dtype=object).T
    ordinal = OrdinalEncoder(min_frequency=4).fit(X)
    X_test = np.array([['snake', 'red'], ['deer', 'green'], [np.nan, 'green'], ['dog', 'green'], ['cat', 'red']], dtype=object)
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, [[2, 1], [2, 0], [np.nan, 0], [1, 0], [0, 1]])