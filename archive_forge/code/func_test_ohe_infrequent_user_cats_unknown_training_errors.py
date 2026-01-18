import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('kwargs', [{'min_frequency': 2, 'max_categories': 3}])
def test_ohe_infrequent_user_cats_unknown_training_errors(kwargs):
    """All user provided categories are infrequent."""
    X_train = np.array([['e'] * 3], dtype=object).T
    ohe = OneHotEncoder(categories=[['c', 'd', 'a', 'b']], sparse_output=False, handle_unknown='infrequent_if_exist', **kwargs).fit(X_train)
    X_trans = ohe.transform([['a'], ['e']])
    assert_allclose(X_trans, [[1], [1]])