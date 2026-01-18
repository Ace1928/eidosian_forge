import re
import numpy as np
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import is_scalar_nan
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_one_hot_encoder_set_params():
    X = np.array([[1, 2]]).T
    oh = OneHotEncoder()
    oh.set_params(categories=[[0, 1, 2, 3]])
    assert oh.get_params()['categories'] == [[0, 1, 2, 3]]
    assert oh.fit_transform(X).toarray().shape == (2, 4)
    oh.set_params(categories=[[0, 1, 2, 3, 4]])
    assert oh.fit_transform(X).toarray().shape == (2, 5)