import numpy
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from thinc.api import Linear, NumpyOps
def W_b_inputs(shape):
    batch_size, nr_out, nr_in = shape
    W = ndarrays_of_shape((nr_out, nr_in))
    b = ndarrays_of_shape((nr_out,))
    input_ = ndarrays_of_shape((batch_size, nr_in))
    return tuples(W, b, input_)