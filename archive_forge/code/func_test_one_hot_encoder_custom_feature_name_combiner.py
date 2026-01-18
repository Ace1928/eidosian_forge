import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_custom_feature_name_combiner():
    """Check the behaviour of `feature_name_combiner` as a callable."""

    def name_combiner(feature, category):
        return feature + '_' + repr(category)
    enc = OneHotEncoder(feature_name_combiner=name_combiner)
    X = np.array([['None', None]], dtype=object).T
    enc.fit(X)
    feature_names = enc.get_feature_names_out()
    assert_array_equal(["x0_'None'", 'x0_None'], feature_names)
    feature_names = enc.get_feature_names_out(input_features=['a'])
    assert_array_equal(["a_'None'", 'a_None'], feature_names)

    def wrong_combiner(feature, category):
        return 0
    enc = OneHotEncoder(feature_name_combiner=wrong_combiner).fit(X)
    err_msg = 'When `feature_name_combiner` is a callable, it should return a Python string.'
    with pytest.raises(TypeError, match=err_msg):
        enc.get_feature_names_out()