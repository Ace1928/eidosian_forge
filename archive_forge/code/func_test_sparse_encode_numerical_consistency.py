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
@pytest.mark.parametrize('algorithm', ('lasso_lars', 'lasso_cd', 'lars', 'threshold', 'omp'))
def test_sparse_encode_numerical_consistency(algorithm):
    rtol = 0.0001
    n_components = 6
    rng = np.random.RandomState(0)
    dictionary = rng.randn(n_components, n_features)
    code_32 = sparse_encode(X.astype(np.float32), dictionary.astype(np.float32), algorithm=algorithm)
    code_64 = sparse_encode(X.astype(np.float64), dictionary.astype(np.float64), algorithm=algorithm)
    assert_allclose(code_32, code_64, rtol=rtol)