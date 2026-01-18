import sys
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse import random as sparse_random
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._csr_polynomial_expansion import (
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('extrapolation', ['constant', 'linear', 'continue', 'periodic'])
@pytest.mark.parametrize('degree', [2, 3])
def test_split_transform_feature_names_extrapolation_degree(extrapolation, degree):
    """Test feature names are correct for different extrapolations and degree.

    Non-regression test for gh-25292.
    """
    X = np.arange(20).reshape(10, 2)
    splt = SplineTransformer(degree=degree, extrapolation=extrapolation).fit(X)
    feature_names = splt.get_feature_names_out(['a', 'b'])
    assert len(feature_names) == splt.n_features_out_
    X_trans = splt.transform(X)
    assert X_trans.shape[1] == len(feature_names)