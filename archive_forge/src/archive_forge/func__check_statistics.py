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
def _check_statistics(X, X_true, strategy, statistics, missing_values, sparse_container):
    """Utility function for testing imputation for a given strategy.

    Test with dense and sparse arrays

    Check that:
        - the statistics (mean, median, mode) are correct
        - the missing values are imputed correctly"""
    err_msg = 'Parameters: strategy = %s, missing_values = %s, sparse = {0}' % (strategy, missing_values)
    assert_ae = assert_array_equal
    if X.dtype.kind == 'f' or X_true.dtype.kind == 'f':
        assert_ae = assert_array_almost_equal
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    X_trans = imputer.fit(X).transform(X.copy())
    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(False))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(False))
    imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)
    imputer.fit(sparse_container(X))
    X_trans = imputer.transform(sparse_container(X.copy()))
    if sparse.issparse(X_trans):
        X_trans = X_trans.toarray()
    assert_ae(imputer.statistics_, statistics, err_msg=err_msg.format(True))
    assert_ae(X_trans, X_true, err_msg=err_msg.format(True))