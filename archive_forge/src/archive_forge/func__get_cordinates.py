import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (NonlinearConstraint,
def _get_cordinates(self, x):
    x_coord = x[:self.n_electrons]
    y_coord = x[self.n_electrons:2 * self.n_electrons]
    z_coord = x[2 * self.n_electrons:]
    return (x_coord, y_coord, z_coord)