import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_valid_slicing(self):
    a = np.array([[[5]]])
    a[:]
    a[0:]
    a[:2]
    a[0:2]
    a[::2]
    a[1::2]
    a[:2:2]
    a[1:2:2]