import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.ensemble._hist_gradient_boosting.common import (
from sklearn.ensemble._hist_gradient_boosting.histogram import HistogramBuilder
from sklearn.ensemble._hist_gradient_boosting.splitting import (
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import skip_if_32bit
def _assert_categories_equals_bitset(categories, bitset):
    expected_bitset = np.zeros(8, dtype=np.uint32)
    for cat in categories:
        idx = cat // 32
        shift = cat % 32
        expected_bitset[idx] |= 1 << shift
    assert_array_equal(expected_bitset, bitset)