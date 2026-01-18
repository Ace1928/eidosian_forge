import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_omp_gram_numerical_consistency():
    coef_32 = orthogonal_mp_gram(G.astype(np.float32), Xy.astype(np.float32), n_nonzero_coefs=5)
    coef_64 = orthogonal_mp_gram(G.astype(np.float32), Xy.astype(np.float64), n_nonzero_coefs=5)
    assert_allclose(coef_32, coef_64)