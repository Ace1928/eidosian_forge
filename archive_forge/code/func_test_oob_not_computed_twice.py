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
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_oob_not_computed_twice(name):
    X, y = (hastie_X, hastie_y)
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=10, warm_start=True, bootstrap=True, oob_score=True)
    with patch.object(est, '_set_oob_score_and_attributes', wraps=est._set_oob_score_and_attributes) as mock_set_oob_score_and_attributes:
        est.fit(X, y)
        with pytest.warns(UserWarning, match='Warm-start fitting without increasing'):
            est.fit(X, y)
        mock_set_oob_score_and_attributes.assert_called_once()