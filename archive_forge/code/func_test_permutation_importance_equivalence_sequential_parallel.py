import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, scale
from sklearn.utils import parallel_backend
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('max_samples', [500, 1.0])
def test_permutation_importance_equivalence_sequential_parallel(max_samples):
    X, y = make_regression(n_samples=500, n_features=10, random_state=0)
    lr = LinearRegression().fit(X, y)
    importance_sequential = permutation_importance(lr, X, y, n_repeats=5, random_state=0, n_jobs=1, max_samples=max_samples)
    imp_min = importance_sequential['importances'].min()
    imp_max = importance_sequential['importances'].max()
    assert imp_max - imp_min > 0.3
    importance_processes = permutation_importance(lr, X, y, n_repeats=5, random_state=0, n_jobs=2)
    assert_allclose(importance_processes['importances'], importance_sequential['importances'])
    with parallel_backend('threading'):
        importance_threading = permutation_importance(lr, X, y, n_repeats=5, random_state=0, n_jobs=2)
    assert_allclose(importance_threading['importances'], importance_sequential['importances'])