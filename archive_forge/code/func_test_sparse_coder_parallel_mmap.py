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
def test_sparse_coder_parallel_mmap():
    rng = np.random.RandomState(777)
    n_components, n_features = (40, 64)
    init_dict = rng.rand(n_components, n_features)
    n_samples = int(2000000.0) // (4 * n_features)
    data = np.random.rand(n_samples, n_features).astype(np.float32)
    sc = SparseCoder(init_dict, transform_algorithm='omp', n_jobs=2)
    sc.fit_transform(data)