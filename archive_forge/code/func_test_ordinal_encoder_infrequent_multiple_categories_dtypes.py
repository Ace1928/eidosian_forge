import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ordinal_encoder_infrequent_multiple_categories_dtypes():
    """Test infrequent categories with a pandas DataFrame with multiple dtypes."""
    pd = pytest.importorskip('pandas')
    categorical_dtype = pd.CategoricalDtype(['bird', 'cat', 'dog', 'snake'])
    X = pd.DataFrame({'str': ['a', 'f', 'c', 'f', 'f', 'a', 'c', 'b', 'b'], 'int': [5, 3, 0, 10, 10, 12, 0, 3, 5], 'categorical': pd.Series(['dog'] * 4 + ['cat'] * 3 + ['snake'] + ['bird'], dtype=categorical_dtype)}, columns=['str', 'int', 'categorical'])
    ordinal = OrdinalEncoder(max_categories=3).fit(X)
    assert_array_equal(ordinal.infrequent_categories_[0], ['a', 'b'])
    assert_array_equal(ordinal.infrequent_categories_[1], [0, 3, 12])
    assert_array_equal(ordinal.infrequent_categories_[2], ['bird', 'snake'])
    X_test = pd.DataFrame({'str': ['a', 'b', 'f', 'c'], 'int': [12, 0, 10, 5], 'categorical': pd.Series(['cat'] + ['snake'] + ['bird'] + ['dog'], dtype=categorical_dtype)}, columns=['str', 'int', 'categorical'])
    expected_trans = [[2, 2, 0], [2, 2, 2], [1, 1, 2], [0, 0, 1]]
    X_trans = ordinal.transform(X_test)
    assert_allclose(X_trans, expected_trans)