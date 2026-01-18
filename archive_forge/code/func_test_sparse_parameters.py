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
@pytest.mark.parametrize('tree_type', SPARSE_TREES)
@pytest.mark.parametrize('dataset', ['sparse-pos', 'sparse-neg', 'sparse-mix', 'zeros'])
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_sparse_parameters(tree_type, dataset, csc_container):
    TreeEstimator = ALL_TREES[tree_type]
    X = DATASETS[dataset]['X']
    X_sparse = csc_container(X)
    y = DATASETS[dataset]['y']
    d = TreeEstimator(random_state=0, max_features=1, max_depth=2).fit(X, y)
    s = TreeEstimator(random_state=0, max_features=1, max_depth=2).fit(X_sparse, y)
    assert_tree_equal(d.tree_, s.tree_, '{0} with dense and sparse format gave different trees'.format(tree_type))
    assert_array_almost_equal(s.predict(X), d.predict(X))
    d = TreeEstimator(random_state=0, max_features=1, min_samples_split=10).fit(X, y)
    s = TreeEstimator(random_state=0, max_features=1, min_samples_split=10).fit(X_sparse, y)
    assert_tree_equal(d.tree_, s.tree_, '{0} with dense and sparse format gave different trees'.format(tree_type))
    assert_array_almost_equal(s.predict(X), d.predict(X))
    d = TreeEstimator(random_state=0, min_samples_leaf=X_sparse.shape[0] // 2).fit(X, y)
    s = TreeEstimator(random_state=0, min_samples_leaf=X_sparse.shape[0] // 2).fit(X_sparse, y)
    assert_tree_equal(d.tree_, s.tree_, '{0} with dense and sparse format gave different trees'.format(tree_type))
    assert_array_almost_equal(s.predict(X), d.predict(X))
    d = TreeEstimator(random_state=0, max_leaf_nodes=3).fit(X, y)
    s = TreeEstimator(random_state=0, max_leaf_nodes=3).fit(X_sparse, y)
    assert_tree_equal(d.tree_, s.tree_, '{0} with dense and sparse format gave different trees'.format(tree_type))
    assert_array_almost_equal(s.predict(X), d.predict(X))