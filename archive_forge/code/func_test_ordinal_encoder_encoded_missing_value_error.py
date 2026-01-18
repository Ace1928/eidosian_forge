import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('with_pandas', [True, False])
def test_ordinal_encoder_encoded_missing_value_error(with_pandas):
    """Check OrdinalEncoder errors when encoded_missing_value is used by
    an known category."""
    X = np.array([['a', 'dog'], ['b', 'cat'], ['c', np.nan]], dtype=object)
    error_msg = 'encoded_missing_value \\(1\\) is already used to encode a known category in features: '
    if with_pandas:
        pd = pytest.importorskip('pandas')
        X = pd.DataFrame(X, columns=['letter', 'pet'])
        error_msg = error_msg + "\\['pet'\\]"
    else:
        error_msg = error_msg + '\\[1\\]'
    oe = OrdinalEncoder(encoded_missing_value=1)
    with pytest.raises(ValueError, match=error_msg):
        oe.fit(X)