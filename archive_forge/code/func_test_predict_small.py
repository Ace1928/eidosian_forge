import numpy
import pytest
from hypothesis import given, settings
from mock import MagicMock
from numpy.testing import assert_allclose
from thinc.api import SGD, Dropout, Linear, chain
from ..strategies import arrays_OI_O_BI
from ..util import get_model, get_shape
@settings(max_examples=100)
@given(arrays_OI_O_BI(max_batch=8, max_out=8, max_in=8))
def test_predict_small(W_b_input):
    W, b, input_ = W_b_input
    nr_out, nr_in = W.shape
    model = Linear(nr_out, nr_in)
    model.set_param('W', W)
    model.set_param('b', b)
    einsummed = numpy.einsum('oi,bi->bo', numpy.asarray(W, dtype='float64'), numpy.asarray(input_, dtype='float64'), optimize=False)
    expected_output = einsummed + b
    predicted_output = model.predict(input_)
    assert_allclose(predicted_output, expected_output, rtol=0.01, atol=0.01)