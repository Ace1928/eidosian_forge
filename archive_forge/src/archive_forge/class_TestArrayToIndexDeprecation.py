import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestArrayToIndexDeprecation:
    """Creating an index from array not 0-D is an error.

    """

    def test_array_to_index_error(self):
        a = np.array([[[1]]])
        assert_raises(TypeError, operator.index, np.array([1]))
        assert_raises(TypeError, np.reshape, a, (a, -1))
        assert_raises(TypeError, np.take, a, [0], a)