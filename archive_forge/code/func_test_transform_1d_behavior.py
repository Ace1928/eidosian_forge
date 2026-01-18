import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
def test_transform_1d_behavior():
    X = np.arange(4)
    est = KBinsDiscretizer(n_bins=2)
    with pytest.raises(ValueError):
        est.fit(X)
    est = KBinsDiscretizer(n_bins=2)
    est.fit(X.reshape(-1, 1))
    with pytest.raises(ValueError):
        est.transform(X)