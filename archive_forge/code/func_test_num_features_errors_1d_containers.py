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
@pytest.mark.parametrize('X', [[1, 2, 3], ['a', 'b', 'c'], [False, True, False], [1.0, 3.4, 4.0], [{'a': 1}, {'b': 2}, {'c': 3}]], ids=['int', 'str', 'bool', 'float', 'dict'])
@pytest.mark.parametrize('constructor_name', ['list', 'tuple', 'array', 'series'])
def test_num_features_errors_1d_containers(X, constructor_name):
    X = _convert_container(X, constructor_name)
    if constructor_name == 'array':
        expected_type_name = 'numpy.ndarray'
    elif constructor_name == 'series':
        expected_type_name = 'pandas.core.series.Series'
    else:
        expected_type_name = constructor_name
    message = f'Unable to find the number of features from X of type {expected_type_name}'
    if hasattr(X, 'shape'):
        message += ' with shape (3,)'
    elif isinstance(X[0], str):
        message += ' where the samples are of type str'
    elif isinstance(X[0], dict):
        message += ' where the samples are of type dict'
    with pytest.raises(TypeError, match=re.escape(message)):
        _num_features(X)