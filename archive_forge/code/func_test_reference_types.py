import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_reference_types():
    input_array = np.array('a', dtype=object)
    expected = np.array(['a'] * 3, dtype=object)
    actual = broadcast_to(input_array, (3,))
    assert_array_equal(expected, actual)
    actual, _ = broadcast_arrays(input_array, np.ones(3))
    assert_array_equal(expected, actual)