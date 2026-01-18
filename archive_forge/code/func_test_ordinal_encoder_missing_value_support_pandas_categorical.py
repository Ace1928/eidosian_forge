import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('pd_nan_type', ['pd.NA', 'np.nan'])
@pytest.mark.parametrize('encoded_missing_value', [np.nan, -2])
def test_ordinal_encoder_missing_value_support_pandas_categorical(pd_nan_type, encoded_missing_value):
    """Check ordinal encoder is compatible with pandas."""
    pd = pytest.importorskip('pandas')
    pd_missing_value = pd.NA if pd_nan_type == 'pd.NA' else np.nan
    df = pd.DataFrame({'col1': pd.Series(['c', 'a', pd_missing_value, 'b', 'a'], dtype='category')})
    oe = OrdinalEncoder(encoded_missing_value=encoded_missing_value).fit(df)
    assert len(oe.categories_) == 1
    assert_array_equal(oe.categories_[0][:3], ['a', 'b', 'c'])
    assert np.isnan(oe.categories_[0][-1])
    df_trans = oe.transform(df)
    assert_allclose(df_trans, [[2.0], [0.0], [encoded_missing_value], [1.0], [0.0]])
    X_inverse = oe.inverse_transform(df_trans)
    assert X_inverse.shape == (5, 1)
    assert_array_equal(X_inverse[:2, 0], ['c', 'a'])
    assert_array_equal(X_inverse[3:, 0], ['b', 'a'])
    assert np.isnan(X_inverse[2, 0])