import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
@pytest.mark.parametrize('compute_z,d,e,d_expect,z_expect', [(2, np.array([4.16, 5.25, 1.09, 0.62]), np.array([3.17, -0.97, 0.55]), np.array([8.0023, 1.9926, 1.0014, 0.1237]), np.array([[0.6326, 0.6245, -0.4191, 0.1847], [0.7668, -0.427, 0.4176, -0.2352], [-0.1082, 0.6071, 0.4594, -0.6393], [-0.0081, 0.2432, 0.6625, 0.7084]]))])
def test_pteqr_NAG_f08jgf(compute_z, d, e, d_expect, z_expect):
    """
    Implements real (f08jgf) example from NAG Manual Mark 26.
    Tests for correct outputs.
    """
    atol = 0.0001
    pteqr = get_lapack_funcs('pteqr', dtype=d.dtype)
    z = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    _d, _e, _z, info = pteqr(d=d, e=e, z=z, compute_z=compute_z)
    assert_allclose(_d, d_expect, atol=atol)
    assert_allclose(np.abs(_z), np.abs(z_expect), atol=atol)