from functools import partial
import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import Linear, NumpyOps, Relu, chain
Helper for gradient check. To do the numeric gradient check, we have
    to be able to wiggle one value in a parameter, and check the prediction
    before and after. So we need to get a callback that gives an output
    given a change to one weight.
    