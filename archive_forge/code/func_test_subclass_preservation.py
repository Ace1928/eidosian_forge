import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_subclass_preservation(self):

    class MinimalSubclass(np.ndarray):
        pass
    self.test_scalar_array(MinimalSubclass)
    self.test_0d_array(MinimalSubclass)
    self.test_axis_insertion(MinimalSubclass)