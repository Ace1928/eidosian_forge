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
def test_random_trees_embedding_feature_names_out():
    """Check feature names out for Random Trees Embedding."""
    random_state = np.random.RandomState(0)
    X = np.abs(random_state.randn(100, 4))
    hasher = RandomTreesEmbedding(n_estimators=2, max_depth=2, sparse_output=False, random_state=0).fit(X)
    names = hasher.get_feature_names_out()
    expected_names = [f'randomtreesembedding_{tree}_{leaf}' for tree, leaf in [(0, 2), (0, 3), (0, 5), (0, 6), (1, 2), (1, 3), (1, 5), (1, 6)]]
    assert_array_equal(expected_names, names)