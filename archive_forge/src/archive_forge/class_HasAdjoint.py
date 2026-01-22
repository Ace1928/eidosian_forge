from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
class HasAdjoint(BaseMatlike):
    args = ()

    def _adjoint(self):
        shape = (self.shape[1], self.shape[0])
        matvec = partial(rmv, dtype=self.dtype)
        rmatvec = partial(mv, dtype=self.dtype)
        return interface.LinearOperator(matvec=matvec, rmatvec=rmatvec, dtype=self.dtype, shape=shape)