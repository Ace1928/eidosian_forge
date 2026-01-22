import inspect
import sys
import pytest
import numpy as np
from numpy.core import arange
from numpy.testing import assert_, assert_equal, assert_raises_regex
from numpy.lib import deprecate, deprecate_with_doc
import numpy.lib.utils as utils
from io import StringIO
class TestByteBounds:

    def test_byte_bounds(self):
        a = arange(12).reshape(3, 4)
        low, high = utils.byte_bounds(a)
        assert_equal(high - low, a.size * a.itemsize)

    def test_unusual_order_positive_stride(self):
        a = arange(12).reshape(3, 4)
        b = a.T
        low, high = utils.byte_bounds(b)
        assert_equal(high - low, b.size * b.itemsize)

    def test_unusual_order_negative_stride(self):
        a = arange(12).reshape(3, 4)
        b = a.T[::-1]
        low, high = utils.byte_bounds(b)
        assert_equal(high - low, b.size * b.itemsize)

    def test_strided(self):
        a = arange(12)
        b = a[::2]
        low, high = utils.byte_bounds(b)
        assert_equal(high - low, b.size * 2 * b.itemsize - b.itemsize)