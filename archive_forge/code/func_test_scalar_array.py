import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_scalar_array(self, cls=np.ndarray):
    a = np.ones((6, 3)).view(cls)
    res = apply_along_axis(np.sum, 0, a)
    assert_(isinstance(res, cls))
    assert_array_equal(res, np.array([6, 6, 6]).view(cls))