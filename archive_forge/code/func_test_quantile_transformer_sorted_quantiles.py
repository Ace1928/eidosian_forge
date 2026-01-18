import re
import warnings
import numpy as np
import numpy.linalg as la
import pytest
from scipy import sparse, stats
from sklearn import datasets
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
from sklearn.preprocessing._data import BOUNDS_THRESHOLD, _handle_zeros_in_scale
from sklearn.svm import SVR
from sklearn.utils import gen_batches, shuffle
from sklearn.utils._array_api import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import (
from sklearn.utils.sparsefuncs import mean_variance_axis
@pytest.mark.parametrize('array_type', ['array', 'sparse'])
def test_quantile_transformer_sorted_quantiles(array_type):
    X = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 1, 1, 9, 9, 9, 8, 8, 7] * 10)
    X = 0.1 * X.reshape(-1, 1)
    X = _convert_container(X, array_type)
    n_quantiles = 100
    qt = QuantileTransformer(n_quantiles=n_quantiles).fit(X)
    quantiles = qt.quantiles_[:, 0]
    assert len(quantiles) == 100
    assert all(np.diff(quantiles) >= 0)