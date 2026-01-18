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
@pytest.mark.parametrize('name', FOREST_REGRESSORS)
@pytest.mark.parametrize('criterion', ('squared_error', 'absolute_error', 'friedman_mse'))
def test_regression_criterion(name, criterion):
    ForestRegressor = FOREST_REGRESSORS[name]
    reg = ForestRegressor(n_estimators=5, criterion=criterion, random_state=1)
    reg.fit(X_reg, y_reg)
    score = reg.score(X_reg, y_reg)
    assert score > 0.93, 'Failed with max_features=None, criterion %s and score = %f' % (criterion, score)
    reg = ForestRegressor(n_estimators=5, criterion=criterion, max_features=6, random_state=1)
    reg.fit(X_reg, y_reg)
    score = reg.score(X_reg, y_reg)
    assert score > 0.92, 'Failed with max_features=6, criterion %s and score = %f' % (criterion, score)