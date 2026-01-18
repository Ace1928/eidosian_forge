import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ohe_infrequent_multiple_categories_dtypes():
    """Test infrequent categories with a pandas dataframe with multiple dtypes."""
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'str': ['a', 'f', 'c', 'f', 'f', 'a', 'c', 'b', 'b'], 'int': [5, 3, 0, 10, 10, 12, 0, 3, 5]}, columns=['str', 'int'])
    ohe = OneHotEncoder(categories='auto', max_categories=3, handle_unknown='infrequent_if_exist')
    X_trans = ohe.fit_transform(X).toarray()
    assert_array_equal(ohe.infrequent_categories_[0], ['a', 'b'])
    assert_array_equal(ohe.infrequent_categories_[1], [0, 3, 12])
    expected = [[0, 0, 1, 1, 0, 0], [0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0]]
    assert_allclose(expected, X_trans)
    X_test = pd.DataFrame({'str': ['b', 'f'], 'int': [14, 12]}, columns=['str', 'int'])
    expected = [[0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 0, 1]]
    X_test_trans = ohe.transform(X_test)
    assert_allclose(expected, X_test_trans.toarray())
    X_inv = ohe.inverse_transform(X_test_trans)
    expected_inv = np.array([['infrequent_sklearn', 'infrequent_sklearn'], ['f', 'infrequent_sklearn']], dtype=object)
    assert_array_equal(expected_inv, X_inv)
    X_test = pd.DataFrame({'str': ['c', 'b'], 'int': [12, 5]}, columns=['str', 'int'])
    X_test_trans = ohe.transform(X_test).toarray()
    expected = [[1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0]]
    assert_allclose(expected, X_test_trans)
    X_inv = ohe.inverse_transform(X_test_trans)
    expected_inv = np.array([['c', 'infrequent_sklearn'], ['infrequent_sklearn', 5]], dtype=object)
    assert_array_equal(expected_inv, X_inv)