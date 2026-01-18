import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_inverse_if_binary():
    X = np.array([['Male', 1], ['Female', 3], ['Female', 2]], dtype=object)
    ohe = OneHotEncoder(drop='if_binary', sparse_output=False)
    X_tr = ohe.fit_transform(X)
    assert_array_equal(ohe.inverse_transform(X_tr), X)