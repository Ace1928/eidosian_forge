import itertools
import math
import pickle
from collections import defaultdict
from functools import partial
from itertools import combinations, product
from typing import Any, Dict
from unittest.mock import patch
import joblib
import numpy as np
import pytest
from scipy.special import comb
import sklearn
from sklearn import clone, datasets
from sklearn.datasets import make_classification, make_hastie_10_2
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._forest import (
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree._classes import SPARSE_SPLITTERS
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel
from sklearn.utils.validation import check_random_state
def test_poisson_vs_mse():
    """Test that random forest with poisson criterion performs better than
    mse for a poisson target.

    There is a similar test for DecisionTreeRegressor.
    """
    rng = np.random.RandomState(42)
    n_train, n_test, n_features = (500, 500, 10)
    X = datasets.make_low_rank_matrix(n_samples=n_train + n_test, n_features=n_features, random_state=rng)
    coef = rng.uniform(low=-2, high=2, size=n_features) / np.max(X, axis=0)
    y = rng.poisson(lam=np.exp(X @ coef))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=rng)
    forest_poi = RandomForestRegressor(criterion='poisson', min_samples_leaf=10, max_features='sqrt', random_state=rng)
    forest_mse = RandomForestRegressor(criterion='squared_error', min_samples_leaf=10, max_features='sqrt', random_state=rng)
    forest_poi.fit(X_train, y_train)
    forest_mse.fit(X_train, y_train)
    dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)
    for X, y, data_name in [(X_train, y_train, 'train'), (X_test, y_test, 'test')]:
        metric_poi = mean_poisson_deviance(y, forest_poi.predict(X))
        metric_mse = mean_poisson_deviance(y, np.clip(forest_mse.predict(X), 1e-06, None))
        metric_dummy = mean_poisson_deviance(y, dummy.predict(X))
        if data_name == 'test':
            assert metric_poi < metric_mse
        assert metric_poi < 0.8 * metric_dummy