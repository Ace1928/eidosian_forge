import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_tile_one_repetition_on_array_gh4679(self):
    a = np.arange(5)
    b = tile(a, 1)
    b += 2
    assert_equal(a, np.arange(5))