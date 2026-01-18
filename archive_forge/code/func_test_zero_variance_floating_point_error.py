import numpy as np
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import BSR_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.skipif(np.var(data2) == 0, reason='This test is not valid for this platform, as it relies on numerical instabilities.')
@pytest.mark.parametrize('sparse_container', [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_zero_variance_floating_point_error(sparse_container):
    X = data2 if sparse_container is None else sparse_container(data2)
    msg = 'No feature in X meets the variance threshold 0.00000'
    with pytest.raises(ValueError, match=msg):
        VarianceThreshold().fit(X)