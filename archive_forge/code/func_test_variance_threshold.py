import numpy as np
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import BSR_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
def test_variance_threshold(sparse_container):
    X = data if sparse_container is None else sparse_container(data)
    X = VarianceThreshold(threshold=0.4).fit_transform(X)
    assert (len(data), 1) == X.shape