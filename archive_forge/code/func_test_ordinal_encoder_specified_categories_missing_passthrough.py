import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('X, X2, cats, cat_dtype', [(np.array([['a', np.nan]], dtype=object).T, np.array([['a', 'b']], dtype=object).T, [np.array(['a', 'd', np.nan], dtype=object)], np.object_), (np.array([['a', np.nan]], dtype=object).T, np.array([['a', 'b']], dtype=object).T, [np.array(['a', 'd', np.nan], dtype=object)], np.object_), (np.array([[2.0, np.nan]], dtype=np.float64).T, np.array([[3.0]], dtype=np.float64).T, [np.array([2.0, 4.0, np.nan])], np.float64)], ids=['object-None-missing-value', 'object-nan-missing_value', 'numeric-missing-value'])
def test_ordinal_encoder_specified_categories_missing_passthrough(X, X2, cats, cat_dtype):
    """Test ordinal encoder for specified categories."""
    oe = OrdinalEncoder(categories=cats)
    exp = np.array([[0.0], [np.nan]])
    assert_array_equal(oe.fit_transform(X), exp)
    assert oe.categories_[0].dtype == cat_dtype
    oe = OrdinalEncoder(categories=cats)
    with pytest.raises(ValueError, match='Found unknown categories'):
        oe.fit(X2)