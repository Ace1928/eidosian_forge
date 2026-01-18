from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
def test_predict_and_begin_update_match(model, model1, model2, input_data):
    model = chain(model1, model2)
    via_predict = model.predict(input_data)
    via_update, _ = model.begin_update(input_data)
    assert_allclose(via_predict, via_update)
    expected = get_expected_predict(input_data, [model1.get_param('W'), model2.get_param('W')], [model1.get_param('b'), model2.get_param('b')])
    assert_allclose(via_update, expected, atol=0.01, rtol=0.0001)