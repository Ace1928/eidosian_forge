import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
class TestMultipleEllipsisError:
    """An index can only have a single ellipsis.

    """

    def test_basic(self):
        a = np.arange(10)
        assert_raises(IndexError, lambda: a[..., ...])
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 2,))
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 3,))