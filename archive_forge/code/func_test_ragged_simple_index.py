import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.types import Pairs, Ragged
def test_ragged_simple_index(ragged, i=1):
    r = ragged[i]
    assert_allclose(r.data, ragged.data[4:6])
    assert_allclose(r.lengths, ragged.lengths[i:i + 1])