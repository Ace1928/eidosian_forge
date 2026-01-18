import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_repeated_axis(self):
    a = np.empty((3, 3, 3))
    assert_raises(ValueError, expand_dims, a, axis=(1, 1))