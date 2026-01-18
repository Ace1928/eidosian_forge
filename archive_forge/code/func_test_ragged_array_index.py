import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.types import Pairs, Ragged
def test_ragged_array_index(ragged):
    arr = numpy.array([2, 1, 4], dtype='i')
    r = ragged[arr]
    assert r.data.shape[0] == ragged.lengths[arr].sum()