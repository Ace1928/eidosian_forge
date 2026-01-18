import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.decomposition import (
from sklearn.decomposition._dict_learning import _update_dict
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.parallel import Parallel
def test_sparse_encode_shapes():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    for algo in ('lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'):
        code = sparse_encode(X, V, algorithm=algo)
        assert code.shape == (n_samples, n_components)