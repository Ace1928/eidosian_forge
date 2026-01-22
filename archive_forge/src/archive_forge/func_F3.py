from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def F3(x):
    A = np.array([[-2, 1, 0.0], [1, -2, 1], [0, 1, -2]])
    b = np.array([1, 2, 3.0])
    return A @ x - b