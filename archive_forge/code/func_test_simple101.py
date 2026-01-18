import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_simple101(self):
    a = np.ones((10, 101), 'd')
    assert_array_equal(apply_along_axis(len, 0, a), len(a) * np.ones(a.shape[1]))