import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_unique_onlist(self):
    data = [1, 1, 1, 2, 2, 3]
    test = unique(data, return_index=True, return_inverse=True)
    assert_(isinstance(test[0], MaskedArray))
    assert_equal(test[0], masked_array([1, 2, 3], mask=[0, 0, 0]))
    assert_equal(test[1], [0, 3, 5])
    assert_equal(test[2], [0, 0, 0, 1, 1, 2])