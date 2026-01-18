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
def test_sparse_coder_estimator_clone():
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    coder = SparseCoder(dictionary=V, transform_algorithm='lasso_lars', transform_alpha=0.001)
    cloned = clone(coder)
    assert id(cloned) != id(coder)
    np.testing.assert_allclose(cloned.dictionary, coder.dictionary)
    assert id(cloned.dictionary) != id(coder.dictionary)
    assert cloned.n_components_ == coder.n_components_
    assert cloned.n_features_in_ == coder.n_features_in_
    data = np.random.rand(n_samples, n_features).astype(np.float32)
    np.testing.assert_allclose(cloned.transform(data), coder.transform(data))