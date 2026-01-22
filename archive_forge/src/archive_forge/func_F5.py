from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def F5(x):
    return pressure_network(x, 4, np.array([0.5, 0.5, 0.5, 0.5]))