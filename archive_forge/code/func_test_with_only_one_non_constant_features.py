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
def test_with_only_one_non_constant_features():
    X = np.hstack([np.array([[1.0], [1.0], [0.0], [0.0]]), np.zeros((4, 1000))])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    for name, TreeEstimator in CLF_TREES.items():
        est = TreeEstimator(random_state=0, max_features=1)
        est.fit(X, y)
        assert est.tree_.max_depth == 1
        assert_array_equal(est.predict_proba(X), np.full((4, 2), 0.5))
    for name, TreeEstimator in REG_TREES.items():
        est = TreeEstimator(random_state=0, max_features=1)
        est.fit(X, y)
        assert est.tree_.max_depth == 1
        assert_array_equal(est.predict(X), np.full((4,), 0.5))