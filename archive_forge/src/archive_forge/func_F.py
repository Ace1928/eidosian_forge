from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def F(x):
    x = np.asarray(x).T
    d = diag([3, 2, 1.5, 1, 0.5])
    c = 0.01
    f = -d @ x - c * float(x.T @ x) * x
    return f