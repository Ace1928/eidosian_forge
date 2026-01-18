import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('zoom', [(1, 1), (3, 5), (8, 2), (8, 8)])
@pytest.mark.parametrize('mode', ['nearest', 'constant', 'wrap', 'reflect', 'mirror', 'grid-wrap', 'grid-mirror', 'grid-constant'])
def test_zoom_by_int_order0(self, zoom, mode):
    x = numpy.array([[0, 1], [2, 3]], dtype=float)
    assert_array_almost_equal(ndimage.zoom(x, zoom, order=0, mode=mode), numpy.kron(x, numpy.ones(zoom)))