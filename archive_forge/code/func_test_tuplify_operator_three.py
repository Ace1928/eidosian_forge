import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_tuplify_operator_three(model1, model2, model3):
    with Model.define_operators({'&': tuplify}):
        model = model1 & model2 & model3
        assert len(model.layers) == 2
        assert len(model.layers[0].layers) == 2