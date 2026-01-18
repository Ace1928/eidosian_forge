import numpy as np
import pytest
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import (
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
@pytest.mark.parametrize('data_type', (np.float32, np.float64))
def test_omp_gram_dtype_match(data_type):
    coef = orthogonal_mp_gram(G.astype(data_type), Xy.astype(data_type), n_nonzero_coefs=5)
    assert coef.dtype == data_type