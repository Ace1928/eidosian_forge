import numbers
import re
import warnings
from itertools import product
from operator import itemgetter
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
from pytest import importorskip
import sklearn
from sklearn._config import config_context
from sklearn._min_dependencies import dependent_packages
from sklearn.base import BaseEstimator
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError, PositiveSpectrumWarning
from sklearn.linear_model import ARDRegression
from sklearn.metrics.tests.test_score_objects import EstimatorWithFit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import _sparse_random_matrix
from sklearn.svm import SVR
from sklearn.utils import (
from sklearn.utils._mocking import (
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import _NotAnArray
from sklearn.utils.fixes import (
from sklearn.utils.validation import (
@pytest.mark.parametrize('input_values', [[0, 1, 0, 1, 0, np.nan], [0, 1, 0, 1, 0, 1]])
def test_pandas_array_returns_ndarray(input_values):
    """Check pandas array with extensions dtypes returns a numeric ndarray.

    Non-regression test for gh-25637.
    """
    pd = importorskip('pandas')
    input_series = pd.array(input_values, dtype='Int32')
    result = check_array(input_series, dtype=None, ensure_2d=False, allow_nd=False, force_all_finite=False)
    assert np.issubdtype(result.dtype.kind, np.floating)
    assert_allclose(result, input_values)