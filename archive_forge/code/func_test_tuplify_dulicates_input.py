import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_tuplify_dulicates_input():
    model = tuplify(noop(), noop())
    ones = numpy.ones([10])
    out = model.predict(ones)
    assert out == (ones, ones)