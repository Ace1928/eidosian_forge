from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
def always_four_ones(x):
    x = np.asarray(x)
    assert_(x.shape == (3,) or x.shape == (3, 1))
    return np.ones(4)