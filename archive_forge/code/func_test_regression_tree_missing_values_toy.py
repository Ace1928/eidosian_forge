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
@pytest.mark.parametrize('X', [np.array([np.nan, 2, np.nan, 4, 5, 6]), np.array([np.nan, np.nan, 3, 4, 5, 6]), np.array([1, 2, 3, 4, np.nan, np.nan]), np.array([1, 2, 3, np.nan, 6, np.nan])])
@pytest.mark.parametrize('criterion', ['squared_error', 'friedman_mse'])
def test_regression_tree_missing_values_toy(X, criterion):
    """Check that we properly handle missing values in regression trees using a toy
    dataset.

    The regression targeted by this test was that we were not reinitializing the
    criterion when it comes to the number of missing values. Therefore, the value
    of the critetion (i.e. MSE) was completely wrong.

    This test check that the MSE is null when there is a single sample in the leaf.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28254
    https://github.com/scikit-learn/scikit-learn/issues/28316
    """
    X = X.reshape(-1, 1)
    y = np.arange(6)
    tree = DecisionTreeRegressor(criterion=criterion, random_state=0).fit(X, y)
    tree_ref = clone(tree).fit(y.reshape(-1, 1), y)
    assert all(tree.tree_.impurity >= 0)
    assert_allclose(tree.tree_.impurity[:2], tree_ref.tree_.impurity[:2])
    leaves_idx = np.flatnonzero((tree.tree_.children_left == -1) & (tree.tree_.n_node_samples == 1))
    assert_allclose(tree.tree_.impurity[leaves_idx], 0.0)