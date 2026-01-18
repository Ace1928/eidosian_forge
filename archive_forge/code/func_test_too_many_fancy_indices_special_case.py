import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_too_many_fancy_indices_special_case(self):
    a = np.ones((1,) * 32)
    assert_raises(IndexError, a.__getitem__, (np.array([0]),) * 32)