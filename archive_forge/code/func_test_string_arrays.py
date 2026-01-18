import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_string_arrays(self):
    """Test two arrays with different shapes are found not equal."""
    a = np.array(['floupi', 'floupa'])
    b = np.array(['floupi', 'floupa'])
    self._test_equal(a, b)
    c = np.array(['floupipi', 'floupa'])
    self._test_not_equal(c, b)