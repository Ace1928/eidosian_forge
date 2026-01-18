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
def test_quantile_transform_nan():
    X = np.array([[np.nan, 0, 0, 1], [np.nan, np.nan, 0, 0.5], [np.nan, 1, 1, 0]])
    transformer = QuantileTransformer(n_quantiles=10, random_state=42)
    transformer.fit_transform(X)
    assert np.isnan(transformer.quantiles_[:, 0]).all()
    assert not np.isnan(transformer.quantiles_[:, 1:]).any()