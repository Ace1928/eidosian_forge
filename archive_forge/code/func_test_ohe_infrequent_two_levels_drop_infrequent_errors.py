import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('drop', [['a'], ['d']])
def test_ohe_infrequent_two_levels_drop_infrequent_errors(drop):
    """Test two levels and dropping any infrequent category removes the
    whole infrequent category."""
    X_train = np.array([['a'] * 5 + ['b'] * 20 + ['c'] * 10 + ['d'] * 3]).T
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False, max_categories=2, drop=drop)
    msg = f'Unable to drop category {drop[0]!r} from feature 0 because it is infrequent'
    with pytest.raises(ValueError, match=msg):
        ohe.fit(X_train)