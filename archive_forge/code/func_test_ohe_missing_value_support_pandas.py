import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_ohe_missing_value_support_pandas():
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'col1': ['dog', 'cat', None, 'cat'], 'col2': np.array([3, 0, 4, np.nan], dtype=float)}, columns=['col1', 'col2'])
    expected_df_trans = np.array([[0, 1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 1]])
    Xtr = check_categorical_onehot(df)
    assert_allclose(Xtr, expected_df_trans)