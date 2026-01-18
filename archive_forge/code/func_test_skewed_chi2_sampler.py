import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_skewed_chi2_sampler():
    c = 0.03
    Y[0, 0] = -c / 2.0
    X_c = (X + c)[:, np.newaxis, :]
    Y_c = (Y + c)[np.newaxis, :, :]
    log_kernel = np.log(X_c) / 2.0 + np.log(Y_c) / 2.0 + np.log(2.0) - np.log(X_c + Y_c)
    kernel = np.exp(log_kernel.sum(axis=2))
    transform = SkewedChi2Sampler(skewedness=c, n_components=1000, random_state=42)
    X_trans = transform.fit_transform(X)
    Y_trans = transform.transform(Y)
    kernel_approx = np.dot(X_trans, Y_trans.T)
    assert_array_almost_equal(kernel, kernel_approx, 1)
    assert np.isfinite(kernel).all(), 'NaNs found in the Gram matrix'
    assert np.isfinite(kernel_approx).all(), 'NaNs found in the approximate Gram matrix'
    Y_neg = Y.copy()
    Y_neg[0, 0] = -c * 2.0
    msg = 'X may not contain entries smaller than -skewedness'
    with pytest.raises(ValueError, match=msg):
        transform.transform(Y_neg)