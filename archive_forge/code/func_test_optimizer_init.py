import numpy
import pytest
from thinc.api import Optimizer, registry
def test_optimizer_init():
    optimizer = Optimizer(learn_rate=0.123, use_averages=False, use_radam=True, L2=0.1, L2_is_weight_decay=False)
    _, gradient = optimizer((0, 'x'), numpy.zeros((1, 2)), numpy.zeros(0))
    assert numpy.array_equal(gradient, numpy.zeros(0))
    W = numpy.asarray([1.0, 0.0, 0.0, 1.0], dtype='f').reshape((4,))
    dW = numpy.asarray([[-1.0, 0.0, 0.0, 1.0]], dtype='f').reshape((4,))
    optimizer((0, 'x'), W, dW)
    optimizer = Optimizer(learn_rate=0.123, beta1=0.1, beta2=0.1)
    optimizer((1, 'x'), W, dW)