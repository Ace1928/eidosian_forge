import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
@pytest.mark.parametrize('index', [True, False, np.array([0])])
@pytest.mark.parametrize('num', [32, 40])
@pytest.mark.parametrize('original_ndim', [1, 32])
def test_too_many_advanced_indices(self, index, num, original_ndim):
    arr = np.ones((1,) * original_ndim)
    with pytest.raises(IndexError):
        arr[(index,) * num]
    with pytest.raises(IndexError):
        arr[(index,) * num] = 1.0