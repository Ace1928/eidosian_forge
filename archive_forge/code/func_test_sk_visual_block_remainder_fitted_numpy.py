import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('remainder', ['passthrough', StandardScaler()])
def test_sk_visual_block_remainder_fitted_numpy(remainder):
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    scaler = StandardScaler()
    ct = ColumnTransformer(transformers=[('scale', scaler, [0, 2])], remainder=remainder)
    ct.fit(X)
    visual_block = ct._sk_visual_block_()
    assert visual_block.names == ('scale', 'remainder')
    assert visual_block.name_details == ([0, 2], [1])
    assert visual_block.estimators == (scaler, remainder)