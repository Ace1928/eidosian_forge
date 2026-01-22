import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
@pytest.fixture(scope='module')
def X_sparse():
    rng = check_random_state(42)
    X = sp.random(60, 55, density=0.2, format='csr', random_state=rng)
    X.data[:] = 1 + np.log(X.data)
    return X