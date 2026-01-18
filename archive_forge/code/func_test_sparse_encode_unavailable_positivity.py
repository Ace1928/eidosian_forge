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
@pytest.mark.parametrize('algo', ['lars', 'omp'])
def test_sparse_encode_unavailable_positivity(algo):
    n_components = 12
    rng = np.random.RandomState(0)
    V = rng.randn(n_components, n_features)
    V /= np.sum(V ** 2, axis=1)[:, np.newaxis]
    err_msg = "Positive constraint not supported for '{}' coding method."
    err_msg = err_msg.format(algo)
    with pytest.raises(ValueError, match=err_msg):
        sparse_encode(X, V, algorithm=algo, positive=True)