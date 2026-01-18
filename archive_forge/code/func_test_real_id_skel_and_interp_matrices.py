import scipy.linalg.interpolative as pymatrixid
import numpy as np
from scipy.linalg import hilbert, svdvals, norm
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg.interpolative import interp_decomp
from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
import sys
@pytest.mark.parametrize('rand,lin_op', [(False, False)])
def test_real_id_skel_and_interp_matrices(self, A, L, eps, rank, rand, lin_op):
    k = rank
    A_or_L = A if not lin_op else L
    idx, proj = pymatrixid.interp_decomp(A_or_L, k, rand=rand)
    P = pymatrixid.reconstruct_interp_matrix(idx, proj)
    B = pymatrixid.reconstruct_skel_matrix(A, k, idx)
    assert_allclose(B, A[:, idx[:k]], rtol=eps, atol=1e-08)
    assert_allclose(B @ P, A, rtol=eps, atol=1e-08)