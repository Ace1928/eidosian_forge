import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_single_non_masked_value_on_axis(self):
    data = [[1.0, 0.0], [0.0, 3.0], [0.0, 0.0]]
    masked_arr = np.ma.masked_equal(data, 0)
    expected = [1.0, 3.0]
    assert_array_equal(np.ma.median(masked_arr, axis=0), expected)