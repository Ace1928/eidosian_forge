import numpy
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from thinc.api import Linear, NumpyOps
def arrays_BI(min_B=1, max_B=10, min_I=1, max_I=100):
    shapes = tuples(lengths(lo=min_B, hi=max_B), lengths(lo=min_I, hi=max_I))
    return shapes.flatmap(ndarrays_of_shape)