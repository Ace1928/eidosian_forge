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
@pytest.mark.parametrize('make_data, Tree', [(datasets.make_regression, DecisionTreeRegressor), (datasets.make_classification, DecisionTreeClassifier)])
def test_sample_weight_non_uniform(make_data, Tree):
    """Check sample weight is correctly handled with missing values."""
    rng = np.random.RandomState(0)
    n_samples, n_features = (1000, 10)
    X, y = make_data(n_samples=n_samples, n_features=n_features, random_state=rng)
    X[rng.choice([False, True], size=X.shape, p=[0.9, 0.1])] = np.nan
    sample_weight = np.ones(X.shape[0])
    sample_weight[::2] = 0.0
    tree_with_sw = Tree(random_state=0)
    tree_with_sw.fit(X, y, sample_weight=sample_weight)
    tree_samples_removed = Tree(random_state=0)
    tree_samples_removed.fit(X[1::2, :], y[1::2])
    assert_allclose(tree_samples_removed.predict(X), tree_with_sw.predict(X))