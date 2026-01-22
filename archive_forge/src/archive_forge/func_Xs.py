import numpy
import pytest
from thinc.api import reduce_first, reduce_last, reduce_max, reduce_mean, reduce_sum
from thinc.types import Ragged
@pytest.fixture
def Xs():
    seqs = [numpy.zeros((10, 8), dtype='f'), numpy.zeros((4, 8), dtype='f')]
    for x in seqs:
        x[0] = 1
        x[1] = 2
        x[-1] = -1
    return seqs