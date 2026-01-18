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
def test_power_transformer_boxcox_strictly_positive_exception():
    pt = PowerTransformer(method='box-cox')
    pt.fit(np.abs(X_2d))
    X_with_negatives = X_2d
    not_positive_message = 'strictly positive'
    with pytest.raises(ValueError, match=not_positive_message):
        pt.transform(X_with_negatives)
    with pytest.raises(ValueError, match=not_positive_message):
        pt.fit(X_with_negatives)
    with pytest.raises(ValueError, match=not_positive_message):
        power_transform(X_with_negatives, method='box-cox')
    with pytest.raises(ValueError, match=not_positive_message):
        pt.transform(np.zeros(X_2d.shape))
    with pytest.raises(ValueError, match=not_positive_message):
        pt.fit(np.zeros(X_2d.shape))
    with pytest.raises(ValueError, match=not_positive_message):
        power_transform(np.zeros(X_2d.shape), method='box-cox')