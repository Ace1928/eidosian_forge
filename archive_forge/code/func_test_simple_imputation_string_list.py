import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('strategy, expected', [('most_frequent', 'b'), ('constant', 'missing_value')])
def test_simple_imputation_string_list(strategy, expected):
    X = [['a', 'b'], ['c', np.nan]]
    X_true = np.array([['a', 'b'], ['c', expected]], dtype=object)
    imputer = SimpleImputer(strategy=strategy)
    X_trans = imputer.fit_transform(X)
    assert_array_equal(X_trans, X_true)