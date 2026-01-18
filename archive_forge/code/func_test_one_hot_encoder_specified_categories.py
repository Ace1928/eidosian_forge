import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('handle_unknown', ['ignore', 'infrequent_if_exist'])
@pytest.mark.parametrize('X, X2, cats, cat_dtype', [(np.array([['a', 'b']], dtype=object).T, np.array([['a', 'd']], dtype=object).T, [['a', 'b', 'c']], np.object_), (np.array([[1, 2]], dtype='int64').T, np.array([[1, 4]], dtype='int64').T, [[1, 2, 3]], np.int64), (np.array([['a', 'b']], dtype=object).T, np.array([['a', 'd']], dtype=object).T, [np.array(['a', 'b', 'c'])], np.object_), (np.array([[None, 'a']], dtype=object).T, np.array([[None, 'b']], dtype=object).T, [[None, 'a', 'z']], object), (np.array([['a', 'b']], dtype=object).T, np.array([['a', np.nan]], dtype=object).T, [['a', 'b', 'z']], object), (np.array([['a', None]], dtype=object).T, np.array([['a', np.nan]], dtype=object).T, [['a', None, 'z']], object)], ids=['object', 'numeric', 'object-string', 'object-string-none', 'object-string-nan', 'object-None-and-nan'])
def test_one_hot_encoder_specified_categories(X, X2, cats, cat_dtype, handle_unknown):
    enc = OneHotEncoder(categories=cats)
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert_array_equal(enc.fit_transform(X).toarray(), exp)
    assert list(enc.categories[0]) == list(cats[0])
    assert enc.categories_[0].tolist() == list(cats[0])
    assert enc.categories_[0].dtype == cat_dtype
    enc = OneHotEncoder(categories=cats)
    with pytest.raises(ValueError, match='Found unknown categories'):
        enc.fit(X2)
    enc = OneHotEncoder(categories=cats, handle_unknown=handle_unknown)
    exp = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert_array_equal(enc.fit(X2).transform(X2).toarray(), exp)