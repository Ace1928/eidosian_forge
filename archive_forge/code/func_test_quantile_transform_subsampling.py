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
def test_quantile_transform_subsampling():
    n_samples = 1000000
    n_quantiles = 1000
    X = np.sort(np.random.sample((n_samples, 1)), axis=0)
    ROUND = 5
    inf_norm_arr = []
    for random_state in range(ROUND):
        transformer = QuantileTransformer(random_state=random_state, n_quantiles=n_quantiles, subsample=n_samples // 10)
        transformer.fit(X)
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        assert inf_norm < 0.01
        inf_norm_arr.append(inf_norm)
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)
    X = sparse.rand(n_samples, 1, density=0.99, format='csc', random_state=0)
    inf_norm_arr = []
    for random_state in range(ROUND):
        transformer = QuantileTransformer(random_state=random_state, n_quantiles=n_quantiles, subsample=n_samples // 10)
        transformer.fit(X)
        diff = np.linspace(0, 1, n_quantiles) - np.ravel(transformer.quantiles_)
        inf_norm = np.max(np.abs(diff))
        assert inf_norm < 0.1
        inf_norm_arr.append(inf_norm)
    assert len(np.unique(inf_norm_arr)) == len(inf_norm_arr)