import re
import sys
from io import StringIO
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_small_sparse_partial_fit(sparse_container):
    X_sparse = sparse_container(Xdigits[:100])
    X = Xdigits[:100].copy()
    rbm1 = BernoulliRBM(n_components=64, learning_rate=0.1, batch_size=10, random_state=9)
    rbm2 = BernoulliRBM(n_components=64, learning_rate=0.1, batch_size=10, random_state=9)
    rbm1.partial_fit(X_sparse)
    rbm2.partial_fit(X)
    assert_almost_equal(rbm1.score_samples(X).mean(), rbm2.score_samples(X).mean(), decimal=0)