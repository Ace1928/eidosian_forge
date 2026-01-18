import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_feature_names_unicode():
    enc = OneHotEncoder()
    X = np.array([['c❤t1', 'dat2']], dtype=object).T
    enc.fit(X)
    feature_names = enc.get_feature_names_out()
    assert_array_equal(['x0_c❤t1', 'x0_dat2'], feature_names)
    feature_names = enc.get_feature_names_out(input_features=['n👍me'])
    assert_array_equal(['n👍me_c❤t1', 'n👍me_dat2'], feature_names)