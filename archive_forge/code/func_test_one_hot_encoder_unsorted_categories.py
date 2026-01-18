import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_unsorted_categories():
    X = np.array([['a', 'b']], dtype=object).T
    enc = OneHotEncoder(categories=[['b', 'a', 'c']])
    exp = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert_array_equal(enc.fit(X).transform(X).toarray(), exp)
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert enc.categories_[0].tolist() == ['b', 'a', 'c']
    assert np.issubdtype(enc.categories_[0].dtype, np.object_)
    X = np.array([[1, 2]]).T
    enc = OneHotEncoder(categories=[[2, 1, 3]])
    msg = 'Unsorted categories are not supported'
    with pytest.raises(ValueError, match=msg):
        enc.fit_transform(X)