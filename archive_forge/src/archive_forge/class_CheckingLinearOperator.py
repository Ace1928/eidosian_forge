import os
import re
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest
from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence
class CheckingLinearOperator(LinearOperator):

    def __init__(self, A):
        self.A = A
        self.dtype = A.dtype
        self.shape = A.shape

    def _matvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.dot(x)

    def _rmatvec(self, x):
        assert_equal(max(x.shape), np.size(x))
        return self.A.T.conjugate().dot(x)