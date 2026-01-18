import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.types import Pairs, Ragged
def test_ragged_starts_ends(ragged):
    starts = ragged._get_starts()
    ends = ragged._get_ends()
    assert list(starts) == [0, 4, 6, 14, 15]
    assert list(ends) == [4, 6, 14, 15, 19]