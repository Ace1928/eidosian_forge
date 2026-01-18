import numpy as np
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import BSR_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_zero_variance_value_error():
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        VarianceThreshold().fit([[0, 1], [0, 1]])