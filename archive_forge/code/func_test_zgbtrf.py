import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def test_zgbtrf(self):
    """Compare zgbtrf  LU factorisation with the LU factorisation result
           of linalg.lu."""
    M, N = shape(self.comp_mat)
    lu_symm_band, ipiv, info = zgbtrf(self.bandmat_comp, self.KL, self.KU)
    u = diag(lu_symm_band[2 * self.KL, :])
    for i in range(self.KL + self.KU):
        u += diag(lu_symm_band[2 * self.KL - 1 - i, i + 1:N], i + 1)
    p_lin, l_lin, u_lin = lu(self.comp_mat, permute_l=0)
    assert_array_almost_equal(u, u_lin)