import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_not_equal_nan(self):
    a = np.array([np.nan])
    b = np.array([np.nan])
    assert_raises(AssertionError, assert_allclose, a, b, equal_nan=False)