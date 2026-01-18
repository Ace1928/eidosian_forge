import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_validateaxis(self):
    assert_raises(TypeError, sputils.validateaxis, (0, 1))
    assert_raises(TypeError, sputils.validateaxis, 1.5)
    assert_raises(ValueError, sputils.validateaxis, 3)
    for axis in (-2, -1, 0, 1, None):
        sputils.validateaxis(axis)