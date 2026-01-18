import copy
import copyreg
import io
import pickle
import struct
from itertools import chain, product
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose
from sklearn import clone, datasets, tree
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import _sparse_random_matrix
from sklearn.tree import (
from sklearn.tree._classes import (
from sklearn.tree._tree import (
from sklearn.tree._tree import Tree as CythonTree
from sklearn.utils import _IS_32BIT, compute_sample_weight
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import check_sample_weights_invariance
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('dataset', sorted(set(DATASETS.keys()) - {'reg_small', 'diabetes'}))
@pytest.mark.parametrize('tree_cls', [DecisionTreeClassifier, ExtraTreeClassifier])
def test_prune_tree_classifier_are_subtrees(dataset, tree_cls):
    dataset = DATASETS[dataset]
    X, y = (dataset['X'], dataset['y'])
    est = tree_cls(max_leaf_nodes=20, random_state=0)
    info = est.cost_complexity_pruning_path(X, y)
    pruning_path = info.ccp_alphas
    impurities = info.impurities
    assert np.all(np.diff(pruning_path) >= 0)
    assert np.all(np.diff(impurities) >= 0)
    assert_pruning_creates_subtree(tree_cls, X, y, pruning_path)