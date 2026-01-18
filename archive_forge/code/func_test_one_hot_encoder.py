import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X', [[['def', 1, 55], ['abc', 2, 55]], np.array([[10, 1, 55], [5, 2, 55]]), np.array([['b', 'A', 'cat'], ['a', 'B', 'cat']], dtype=object), np.array([['b', 1, 'cat'], ['a', np.nan, 'cat']], dtype=object), np.array([['b', 1, 'cat'], ['a', float('nan'), 'cat']], dtype=object), np.array([[None, 1, 'cat'], ['a', 2, 'cat']], dtype=object), np.array([[None, 1, None], ['a', np.nan, None]], dtype=object), np.array([[None, 1, None], ['a', float('nan'), None]], dtype=object)], ids=['mixed', 'numeric', 'object', 'mixed-nan', 'mixed-float-nan', 'mixed-None', 'mixed-None-nan', 'mixed-None-float-nan'])
def test_one_hot_encoder(X):
    Xtr = check_categorical_onehot(np.array(X)[:, [0]])
    assert_allclose(Xtr, [[0, 1], [1, 0]])
    Xtr = check_categorical_onehot(np.array(X)[:, [0, 1]])
    assert_allclose(Xtr, [[0, 1, 1, 0], [1, 0, 0, 1]])
    Xtr = OneHotEncoder(categories='auto').fit_transform(X)
    assert_allclose(Xtr.toarray(), [[0, 1, 1, 0, 1], [1, 0, 0, 1, 1]])