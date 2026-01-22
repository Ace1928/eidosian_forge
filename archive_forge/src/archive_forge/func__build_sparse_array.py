import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.linalg import block_diag
from scipy.special import psi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._online_lda_fast import (
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def _build_sparse_array(csr_container):
    n_components = 3
    block = np.full((3, 3), n_components, dtype=int)
    blocks = [block] * n_components
    X = block_diag(*blocks)
    X = csr_container(X)
    return (n_components, X)