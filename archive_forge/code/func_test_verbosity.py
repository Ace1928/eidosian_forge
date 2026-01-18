import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong
def test_verbosity():
    """Check that nonzero verbosity level code runs.
    """
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((10, 10))
    A = X @ X.T
    Q = rnd.standard_normal((X.shape[0], 1))
    msg = 'Exited at iteration.*|Exited postprocessing with accuracies.*'
    with pytest.warns(UserWarning, match=msg):
        _, _ = lobpcg(A, Q, maxiter=3, verbosityLevel=9)