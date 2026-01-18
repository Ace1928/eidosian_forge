import os
import numpy as np
from numpy.testing import (
def test_unique_zero_sized(self):
    assert_array_equal([], np.unique(np.array([])))