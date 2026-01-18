import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.types import Pairs, Ragged
def test_ragged_empty():
    data = numpy.zeros((0, 4), dtype='f')
    lengths = numpy.array([], dtype='i')
    ragged = Ragged(data, lengths)
    assert_allclose(ragged[0:0].data, ragged.data)
    assert_allclose(ragged[0:0].lengths, ragged.lengths)
    assert_allclose(ragged[0:2].data, ragged.data)
    assert_allclose(ragged[0:2].lengths, ragged.lengths)
    assert_allclose(ragged[1:2].data, ragged.data)
    assert_allclose(ragged[1:2].lengths, ragged.lengths)