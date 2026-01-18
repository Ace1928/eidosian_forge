import pickle
import re
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('method', ['transform', 'fit_transform', 'fit'])
def test_routing_passed_metadata_not_supported(method):
    """Test that the right error message is raised when metadata is passed while
    not supported when `enable_metadata_routing=False`."""
    X = np.array([[0, 1, 2], [2, 4, 6]]).T
    y = [1, 2, 3]
    trs = ColumnTransformer([('trans', Trans(), [0])]).fit(X, y)
    with pytest.raises(ValueError, match='is only supported if enable_metadata_routing=True'):
        getattr(trs, method)([[1]], sample_weight=[1], prop='a')