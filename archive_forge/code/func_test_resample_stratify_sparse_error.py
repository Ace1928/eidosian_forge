import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_resample_stratify_sparse_error(csr_container):
    rng = np.random.RandomState(0)
    n_samples = 100
    X = rng.normal(size=(n_samples, 2))
    y = rng.randint(0, 2, size=n_samples)
    stratify = csr_container(y.reshape(-1, 1))
    with pytest.raises(TypeError, match='Sparse data was passed'):
        X, y = resample(X, y, n_samples=50, random_state=rng, stratify=stratify)