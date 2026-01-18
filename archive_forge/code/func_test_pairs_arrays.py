import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.types import Pairs, Ragged
def test_pairs_arrays():
    one = numpy.zeros((128, 45), dtype='f')
    two = numpy.zeros((128, 12), dtype='f')
    pairs = Pairs(one, two)
    assert pairs[:2].one.shape == (2, 45)
    assert pairs[0].two.shape == (12,)
    assert pairs[-1:].one.shape == (1, 45)
    assert pairs[-1:].two.shape == (1, 12)