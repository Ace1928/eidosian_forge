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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_read_only_buffer(csr_container, monkeypatch):
    """RandomForestClassifier must work on readonly sparse data.

    Non-regression test for: https://github.com/scikit-learn/scikit-learn/issues/25333
    """
    monkeypatch.setattr(sklearn.ensemble._forest, 'Parallel', partial(Parallel, max_nbytes=100))
    rng = np.random.RandomState(seed=0)
    X, y = make_classification(n_samples=100, n_features=200, random_state=rng)
    X = csr_container(X, copy=True)
    clf = RandomForestClassifier(n_jobs=2, random_state=rng)
    cross_val_score(clf, X, y, cv=2)