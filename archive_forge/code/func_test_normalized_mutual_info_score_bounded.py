import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.metrics.cluster import (
from sklearn.metrics.cluster._supervised import _generalized_average, check_clusterings
from sklearn.utils import assert_all_finite
from sklearn.utils._testing import assert_almost_equal
@pytest.mark.parametrize('average_method', ['min', 'arithmetic', 'geometric', 'max'])
def test_normalized_mutual_info_score_bounded(average_method):
    """Check that nmi returns a score between 0 (included) and 1 (excluded
    for non-perfect match)

    Non-regression test for issue #13836
    """
    labels1 = [0] * 469
    labels2 = [1] + labels1[1:]
    labels3 = [0, 1] + labels1[2:]
    nmi = normalized_mutual_info_score(labels1, labels2, average_method=average_method)
    assert nmi == 0
    nmi = normalized_mutual_info_score(labels2, labels3, average_method=average_method)
    assert 0 <= nmi < 1