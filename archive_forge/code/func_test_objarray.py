import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_objarray(self):
    """Test object arrays."""
    a = np.array([1, 1], dtype=object)
    self._test_equal(a, 1)