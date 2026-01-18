import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_skewed_chi2_sampler_fitted_attributes_dtype(global_dtype):
    """Check that the fitted attributes are stored accordingly to the
    data type of X."""
    skewed_chi2_sampler = SkewedChi2Sampler()
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=global_dtype)
    skewed_chi2_sampler.fit(X)
    assert skewed_chi2_sampler.random_offset_.dtype == global_dtype
    assert skewed_chi2_sampler.random_weights_.dtype == global_dtype