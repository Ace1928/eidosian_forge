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
def test_sample_hiddens():
    rng = np.random.RandomState(0)
    X = Xdigits[:100]
    rbm1 = BernoulliRBM(n_components=2, batch_size=5, n_iter=5, random_state=42)
    rbm1.fit(X)
    h = rbm1._mean_hiddens(X[0])
    hs = np.mean([rbm1._sample_hiddens(X[0], rng) for i in range(100)], 0)
    assert_almost_equal(h, hs, decimal=1)