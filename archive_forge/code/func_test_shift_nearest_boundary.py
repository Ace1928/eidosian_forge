import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('order', range(0, 6))
@pytest.mark.parametrize('prefilter', [False, True])
def test_shift_nearest_boundary(self, order, prefilter):
    x = numpy.arange(16)
    kwargs = dict(mode='nearest', order=order, prefilter=prefilter)
    assert_array_almost_equal(ndimage.shift(x, order // 2 + 1, **kwargs)[0], x[0])
    assert_array_almost_equal(ndimage.shift(x, -order // 2 - 1, **kwargs)[-1], x[-1])