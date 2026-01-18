import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.types import Pairs, Ragged
def test_ragged_slice_index(ragged, start=0, end=2):
    r = ragged[start:end]
    size = ragged.lengths[start:end].sum()
    assert r.data.shape == (size, r.data.shape[1])
    assert_allclose(r.lengths, ragged.lengths[start:end])