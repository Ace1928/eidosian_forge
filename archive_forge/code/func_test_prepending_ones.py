import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_prepending_ones(self):
    a = np.zeros((3, 2))
    a[...] = np.ones((1, 3, 2))
    a[[0, 1, 2], :] = np.ones((1, 3, 2))
    a[:, [0, 1]] = np.ones((1, 3, 2))
    a[[[0], [1], [2]], [0, 1]] = np.ones((1, 3, 2))