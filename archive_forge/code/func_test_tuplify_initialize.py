import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_tuplify_initialize(nI, nO):
    linear = Linear(nO)
    model = tuplify(linear, linear)
    ones = numpy.ones((1, nI), dtype='float')
    model.initialize(X=ones)