import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('missing_value', [np.nan, None])
def test_ohe_missing_values_get_feature_names(missing_value):
    X = np.array([['a', 'b', missing_value, 'a', missing_value]], dtype=object).T
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(X)
    names = ohe.get_feature_names_out()
    assert_array_equal(names, ['x0_a', 'x0_b', f'x0_{missing_value}'])