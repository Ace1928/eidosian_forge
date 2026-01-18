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
def test_numerical_stability():
    X = np.array([[152.08097839, 140.40744019, 129.75102234, 159.90493774], [142.50700378, 135.8193512, 117.82884979, 162.7578125], [127.28772736, 140.40744019, 129.75102234, 159.90493774], [132.37025452, 143.71923828, 138.35694885, 157.84558105], [103.10237122, 143.71928406, 138.35696411, 157.84559631], [127.71276855, 143.71923828, 138.35694885, 157.84558105], [120.91514587, 140.40744019, 129.75102234, 159.90493774]])
    y = np.array([1.0, 0.70209277, 0.53896582, 0.0, 0.90914464, 0.48026916, 0.49622521])
    with np.errstate(all='raise'):
        for name, Tree in REG_TREES.items():
            reg = Tree(random_state=0)
            reg.fit(X, y)
            reg.fit(X, -y)
            reg.fit(-X, y)
            reg.fit(-X, -y)