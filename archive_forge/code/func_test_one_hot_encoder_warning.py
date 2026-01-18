import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_warning():
    enc = OneHotEncoder()
    X = [['Male', 1], ['Female', 3]]
    np.testing.assert_no_warnings(enc.fit_transform, X)