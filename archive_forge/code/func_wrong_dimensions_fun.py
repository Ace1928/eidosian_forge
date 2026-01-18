import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def wrong_dimensions_fun(self, x):
    return np.array([x ** 2, np.tan(x), np.exp(x)])