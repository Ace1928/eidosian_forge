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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_perplexity_input_format(csr_container):
    n_components, X = _build_sparse_array(csr_container)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=1, learning_method='batch', total_samples=100, random_state=0)
    lda.fit(X)
    perp_1 = lda.perplexity(X)
    perp_2 = lda.perplexity(X.toarray())
    assert_almost_equal(perp_1, perp_2)