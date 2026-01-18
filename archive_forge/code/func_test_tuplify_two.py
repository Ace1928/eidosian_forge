import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_tuplify_two(model1, model2):
    model = tuplify(model1, model2)
    assert len(model.layers) == 2