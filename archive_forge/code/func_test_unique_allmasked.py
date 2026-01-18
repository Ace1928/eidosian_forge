import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_unique_allmasked(self):
    data = masked_array([1, 1, 1], mask=True)
    test = unique(data, return_index=True, return_inverse=True)
    assert_equal(test[0], masked_array([1], mask=[True]))
    assert_equal(test[1], [0])
    assert_equal(test[2], [0, 0, 0])
    data = masked
    test = unique(data, return_index=True, return_inverse=True)
    assert_equal(test[0], masked_array(masked))
    assert_equal(test[1], [0])
    assert_equal(test[2], [0])