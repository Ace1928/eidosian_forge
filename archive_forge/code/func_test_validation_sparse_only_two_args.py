import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from scipy.stats.contingency import crosstab
def test_validation_sparse_only_two_args():
    with pytest.raises(ValueError, match='only two input sequences'):
        crosstab([0, 1, 1], [8, 8, 9], [1, 3, 3], sparse=True)